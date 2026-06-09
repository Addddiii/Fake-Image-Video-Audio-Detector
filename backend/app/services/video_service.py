from fastapi import HTTPException, UploadFile

from app.config import UPLOAD_FOLDER

video_detector = None


def set_video_detector(detector):
    """
    Store the loaded video detector so it can be reused by the API.
    """
    global video_detector
    video_detector = detector


async def analyse_video(file: UploadFile):
    """
    Save the uploaded video temporarily, run prediction, then delete the file.
    """
    file_path = UPLOAD_FOLDER / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if video_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Video model not loaded.",
            )

        prediction = video_detector.predict(str(file_path))
        return prediction

    finally:
        if file_path.exists():
            file_path.unlink()