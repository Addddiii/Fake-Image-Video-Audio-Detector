"""
Fake Media Detection API.
"""

import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import initialize_firebase
from app.config import (
    ALLOWED_ORIGINS,
    AUDIO_MODEL_PATH,
    IMAGE_MODEL_PATH,
    VIDEO_MODEL_PATH,
)
from app.routes.audio_routes import router as audio_router
from app.routes.auth_routes import router as auth_router
from app.routes.image_routes import router as image_router
from app.routes.video_routes import router as video_router
from app.services.audio_service import set_audio_detector
from app.services.video_service import set_video_detector
from inference.audio import load_audio_detector
from inference.image import get_model, initialize_model
from inference.video import load_video_detector


warnings.filterwarnings("ignore", category=FutureWarning)


app = FastAPI(
    title="Fake Media Detection API",
    description="Backend for image, video, and audio fake media detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(image_router)
app.include_router(video_router)
app.include_router(audio_router)


@app.on_event("startup")
async def startup_event():
    initialize_firebase()

    initialize_model(IMAGE_MODEL_PATH)

    try:
        video_detector = load_video_detector(model_path=VIDEO_MODEL_PATH)
        set_video_detector(video_detector)
    except Exception:
        set_video_detector(None)

    try:
        audio_detector = load_audio_detector(model_path=AUDIO_MODEL_PATH)
        set_audio_detector(audio_detector)
    except Exception:
        set_audio_detector(None)


@app.get("/")
def read_root():
    return {
        "message": "Fake Media Detection Backend",
        "status": "running",
    }


@app.get("/health")
def health_check():
    from app.services.audio_service import audio_detector
    from app.services.video_service import video_detector

    return {
        "status": "ok",
        "models": {
            "image": get_model() is not None,
            "video": video_detector is not None,
            "audio": audio_detector is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)