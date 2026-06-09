from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.video_service import analyse_video

router = APIRouter(prefix="/predict", tags=["Video"])


@router.post("/video")
async def predict_video_route(file: UploadFile = File(...)):
    """
    Validate the uploaded video file and return the model prediction.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Only video files are supported.",
        )

    prediction = await analyse_video(file)

    return {"prediction": prediction}