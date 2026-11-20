
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="NeonClipping Backend")

MAX_SIZE = 100 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".mkv", ".webm"}


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
                    raise HTTPException(413, "Video must be under 100 MB")
                out.write(chunk)

        await file.close()

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac",
                "-f", "segment", "-segment_time", "30",
                "-reset_timestamps", "1",
                os.path.join(output_dir, "clip_%03d.mp4")
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        clips = list(Path(output_dir).glob("*.mp4"))
        if result.returncode != 0 or not clips:
            raise HTTPException(500, "Video processing failed")

        zip_path = os.path.join(folder, "neonclipping_clips.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            for clip in clips:
                archive.write(clip, arcname=clip.name)

        return FileResponse(
            zip_path,
            filename="neonclipping_clips.zip",
            media_type="application/zip",
            background=BackgroundTask(
                shutil.rmtree, folder, ignore_errors=True
            )
        )

    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, "Processing error")
      
