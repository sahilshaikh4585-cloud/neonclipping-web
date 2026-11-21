
import os
import shutil
import zipfile
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def get_model():
    global model
    if model is None:
        model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )
    return model


def run_ffmpeg(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2500:])
    return result


def ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def make_captions(audio_path, ass_path):
    whisper = get_model()

    segments, info = whisper.transcribe(
        str(audio_path),
        beam_size=1,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False
    )

    segments = list(segments)
    words = []

    for segment in segments:
        if segment.words:
            for word in segment.words:
                if word.word.strip():
                    words.append(word)

    if not words:
        raise RuntimeError(
            "Voice detect nahi hui. Clear voice wali video se dobara try karein."
        )

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,62,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]

    for i in range(0, len(words), 6):
        group = words[i:i + 6]
        start = group[0].start
        end = max(group[-1].end, start + 0.3)

        caption = ""
        for word in group:
            duration = max(
                1,
                int((word.end - word.start) * 100)
            )
            safe_word = (
                word.word.replace("{", "(")
                .replace("}", ")")
                .replace("\\", "")
            )
            caption += f"{{\\kf{duration}}}{safe_word} "

        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},"
            f"Default,,0,0,0,,{caption.strip()}\n"
        )

    ass_path.write_text("".join(lines), encoding="utf-8")
    return info.language


@app.get("/")
def home():
    return {"status": "NeonClipping backend is running"}


@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    workdir = Path(tempfile.mkdtemp(prefix="neon_"))

    try:
        input_path = workdir / "input.mp4"
        audio_path = workdir / "audio.wav"
        caption_path = workdir / "captions.ass"
        captioned_path = workdir / "captioned.mp4"
        output_dir = workdir / "clips"
        output_dir.mkdir()

        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if input_path.stat().st_size == 0:
            raise HTTPException(400, "Uploaded video is empty.")

        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(audio_path)
        ])

        language = make_captions(audio_path, caption_path)

        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"subtitles={caption_path}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(captioned_path)
        ])

        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(captioned_path),
            "-map", "0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "30",
            "-reset_timestamps", "1",
            str(output_dir / "clip_%03d.mp4")
        ])

        zip_path = workdir / "NeonClipping_Clips.zip"

        clips = list(output_dir.glob("*.mp4"))
        if not clips:
            raise RuntimeError("FFmpeg could not create video clips.")

        with zipfile.ZipFile(zip_path, "w") as archive:
            for clip in clips:
                archive.write(clip, arcname=clip.name)

        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename="NeonClipping_Clips.zip",
            background=BackgroundTask(
                shutil.rmtree, workdir, ignore_errors=True
            )
        )

    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
