
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import uvicorn
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeonClipping Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 100 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".mkv", ".webm"}

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


def ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def make_captions(video_path, ass_path):
    whisper = get_model()

    segments, info = whisper.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True
    )

    highlighted = {
        "wow", "wait", "no", "yes", "crazy",
        "insane", "amazing", "never", "secret",
        "win", "winner", "stop", "what"
    }

    events = []

    for segment in segments:
        words = list(segment.words or [])
        if not words:
            continue

        caption_parts = []

        for word in words:
            text = word.word.strip()
            if not text:
                continue

            text = (
                text.replace("\\", "\\\\")
                .replace("{", "(")
                .replace("}", ")")
            )

            duration = max(
                1,
                round((word.end - word.start) * 100)
            )

            if text.lower().strip(".,!?") in highlighted:
                part = (
                    r"{\c&H0000FFFF&\kf" +
                    str(duration) + "}" +
                    text +
                    r"{\c&H00FFFFFF&}"
                )
            else:
                part = r"{\kf" + str(duration) + "}" + text

            caption_parts.append(part)

        if caption_parts:
            dialogue = (
                f"Dialogue: 0,"
                f"{ass_time(segment.start)},"
                f"{ass_time(segment.end)},"
                f"Default,,0,0,0,,"
                + " ".join(caption_parts)
            )
            events.append(dialogue)

    if not events:
        raise RuntimeError("No speech detected in video")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("""[Script Info]
Title: NeonClipping Animated Captions
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,58,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,1,4,2,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for event in events:
            f.write(event + "\n")


@app.get("/")
def home():
    return {"message": "NeonClipping backend is running"}


@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()

    if ext not in ALLOWED:
        raise HTTPException(400, "Unsupported video format")

    folder = tempfile.mkdtemp(prefix="neon_")
    input_path = os.path.join(folder, "input" + ext)
    output_dir = os.path.join(folder, "clips")
    os.makedirs(output_dir)

    try:
        size = 0

        with open(input_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_SIZE:
                    raise HTTPException(
                        413,
                        "Video must be under 100 MB"
                    )

                out.write(chunk)

        await file.close()

        ass_path = os.path.join(folder, "captions.ass")

        make_captions(input_path, ass_path)

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", f"subtitles={ass_path}",
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-c:a", "aac",
                "-f", "segment",
                "-segment_time", "30",
                "-reset_timestamps", "1",
                os.path.join(output_dir, "clip_%03d.mp4")
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        clips = list(Path(output_dir).glob("*.mp4"))

        if result.returncode != 0 or not clips:
            raise HTTPException(
                500,
                "Caption rendering or video processing failed"
            )

        zip_path = os.path.join(
            folder,
            "neonclipping_clips.zip"
        )

        with zipfile.ZipFile(zip_path, "w") as archive:
            for clip in clips:
                archive.write(clip, arcname=clip.name)

        return FileResponse(
            zip_path,
            filename="neonclipping_clips.zip",
            media_type="application/zip",
            background=BackgroundTask(
                shutil.rmtree,
                folder,
                ignore_errors=True
            )
        )

    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    except Exception as error:
        print("Processing error:", repr(error))
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, "Processing error")

    finally:
        await file.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
