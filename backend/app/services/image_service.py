from fastapi import HTTPException, UploadFile

from app.config import UPLOAD_FOLDER
from inference.image import get_model, predict_image


async def analyse_image(file: UploadFile):
    file_path = UPLOAD_FOLDER / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if get_model() is None:
            raise HTTPException(status_code=503, detail="Image model not loaded.")

        return predict_image(str(file_path))

    finally:
        if file_path.exists():
            file_path.unlink()