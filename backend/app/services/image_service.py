from fastapi import HTTPException, UploadFile

from app.config import UPLOAD_FOLDER

image_detector = None


def set_image_detector(detector):
    """
    Store the loaded image detector so it can be reused by the API.
    """
    global image_detector
    image_detector = detector


async def analyse_image(file: UploadFile):
    """
    Save the uploaded image temporarily, run prediction, then delete the file.
    """
    file_path = UPLOAD_FOLDER / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if image_detector is None:
            raise HTTPException(
                status_code=503,
                detail="Image model not loaded.",
            )

        prediction = image_detector.predict(str(file_path))
        return prediction

    finally:
        if file_path.exists():
            file_path.unlink()