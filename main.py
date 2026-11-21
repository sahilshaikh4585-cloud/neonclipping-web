
import os
import uuid
import shutil
import zipfile
import tempfile
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

app = FastAPI(title="NeonClipping Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
lock = threading.Lock()
model = None


def update_job(job_id, **values):
    with lock:
        if job_id in jobs:
            jobs[job_id].update(values)


def get_model():
    global model
    if model is None:
        model = WhisperModel("small", device="cpu", compute_type="int8")
    return model


def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def seconds_to_ass(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02}:{secs:05.2f}"


def make_ass(segments, start, end, output):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,62,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,1,3,1,5,50,50,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    for segment in segments:
        words = getattr(segment, "words", None) or []
        for i in range(0, len(words), 5):
            group = words[i:i + 5]
            if not group:
                continue

            word_start = max(group[0].start, start)
            word_end = min(group[-1].end, end)

            if word_end <= word_start:
                continue

            text = " ".join(w.word.strip() for w in group)
            text = text.replace("{", "").replace("}", "")
            lines.append(
                f"Dialogue: 0,{seconds_to_ass(word_start-start)},"
                f"{seconds_to_ass(word_end-start)},Default,,0,0,0,,{text}\n"
            )

    Path(output).write_text("".join(lines), encoding="utf-8")


def run_ffmpeg(command, job_id, base, duration, label):
    command += ["-progress", "pipe:1", "-nostats"]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        if line.startswith("out_time_ms="):
            try:
                current = int(line.split("=")[1]) / 1_000_000
                percent = min(1, current / max(duration, 0.1))
                progress = int(base + percent * 35)
                update_job(job_id, progress=progress, status=label)
            except (ValueError, ZeroDivisionError):
                pass

    error = process.stderr.read()
    code = process.wait()

    if code != 0:
        raise RuntimeError(error[-1500:] or "FFmpeg processing failed")


def process_job(job_id, input_path, work_dir):
    try:
        duration = get_duration(input_path)
        if duration <= 0:
            raise RuntimeError("Could not read video duration")

        audio_path = os.path.join(work_dir, "audio.wav")

        update_job(job_id, progress=3, status="Extracting audio")

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-ac", "1", "-ar", "16000",
                audio_path
            ],
            capture_output=True, check=True
        )

        update_job(job_id, progress=10, status="Detecting speech")

        whisper = get_model()
        segments_generator, info = whisper.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False
        )

        segments = []
        for segment in segments_generator:
            segments.append(segment)
            percent = min(1, segment.end / max(duration, 0.1))
            update_job(
                job_id,
                progress=int(10 + percent * 40),
                status="Transcribing speech"
            )

        if not segments:
            raise RuntimeError("No speech detected in video")

        clips_dir = os.path.join(work_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        clip_length = 30
        total_clips = int((duration + clip_length - 1) // clip_length)

        for index in range(total_clips):
            start = index * clip_length
            end = min(start + clip_length, duration)
            length = end - start

            ass_path = os.path.join(work_dir, f"captions_{index}.ass")
            make_ass(segments, start, end, ass_path)

            output_path = os.path.join(
                clips_dir, f"neon_clip_{index + 1}.mp4"
            )

            update_job(
                job_id,
                status=f"Rendering clip {index + 1} of {total_clips}"
            )

            run_ffmpeg(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", input_path,
                    "-t", str(length),
                    "-vf", f"ass={ass_path}",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    output_path
                ],
                job_id,
                50 + (index / total_clips) * 35,
                length,
                f"Rendering clip {index + 1} of {total_clips}"
            )

        update_job(job_id, progress=90, status="Creating ZIP file")

        zip_path = os.path.join(work_dir, "neonclipping_clips.zip")
        files = list(Path(clips_dir).glob("*.mp4"))
        total_bytes = sum(f.stat().st_size for f in files)
        written = 0

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, arcname=file.name)
                written += file.stat().st_size
                percent = written / max(total_bytes, 1)
                update_job(
                    job_id,
                    progress=int(90 + percent * 9),
                    status="Creating ZIP file"
                )

        update_job(
            job_id,
            progress=100,
            status="Completed",
            state="completed",
            download=zip_path
        )

    except Exception as error:
        update_job(
            job_id,
            state="failed",
            status="Processing failed",
            error=str(error)
        )


@app.get("/")
def home():
    return {"message": "NeonClipping backend is running"}


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix="neon_")
    input_path = os.path.join(work_dir, "input_video")

    try:
        with open(input_path, "wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Upload failed")

    jobs[job_id] = {
        "progress": 0,
        "status": "Video uploaded",
        "state": "processing",
        "error": None,
        "download": None
    }

    background_tasks.add_task(process_job, job_id, input_path, work_dir)

    return {"job_id": job_id}


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    with lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "progress": job["progress"],
        "status": job["status"],
        "state": job["state"],
        "error": job["error"]
    }


@app.get("/download/{job_id}")
def download_result(job_id: str):
    with lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["state"] != "completed" or not job["download"]:
        raise HTTPException(status_code=409, detail="Video is not ready")

    return FileResponse(
        job["download"],
        filename="neonclipping_clips.zip",
        media_type="application/zip"
        )
    
