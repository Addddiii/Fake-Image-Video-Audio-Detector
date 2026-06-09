from fastapi import HTTPException, UploadFile

from app.config import UPLOAD_FOLDER

audio_detector = None


def set_audio_detector(detector):
    """
    Store the loaded audio detector so it can be reused by the API.
    """
    global audio_detector
    audio_detector = detector


async def analyse_audio(file: UploadFile):
    """
    Save the uploaded audio file temporarily, run prediction, then delete the file.
    """
    file_path = UPLOAD_FOLDER / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if audio_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Audio model not loaded.",
            )

        prediction = audio_detector.predict(str(file_path))
        return prediction

    finally:
        if file_path.exists():
            file_path.unlink()