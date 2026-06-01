from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", str(BASE_DIR / "models" / "image_model.pth"))
VIDEO_MODEL_PATH = os.getenv("VIDEO_MODEL_PATH", str(BASE_DIR / "models" / "video_model.pth"))
AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", str(BASE_DIR / "models" / "audio_model.pth"))

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
]