from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.image_service import analyse_image


router = APIRouter(prefix="/predict", tags=["Image"])


@router.post("/image")
async def predict_image_route(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    result = await analyse_image(file)
    return {"prediction": result}