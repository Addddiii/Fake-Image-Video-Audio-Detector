from fastapi import HTTPException, UploadFile

from app.config import UPLOAD_FOLDER


audio_detector = None


def set_audio_detector(detector):
    global audio_detector
    audio_detector = detector


async def analyse_audio(file: UploadFile):
    file_path = UPLOAD_FOLDER / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if audio_detector is None:
            raise HTTPException(status_code=503, detail="Audio model not loaded.")

        return audio_detector.predict(str(file_path))

    finally:
        if file_path.exists():
            file_path.unlink()