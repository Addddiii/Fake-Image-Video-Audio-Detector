from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.audio_service import analyse_audio

router = APIRouter(prefix="/predict", tags=["Audio"])


@router.post("/audio")
async def predict_audio_route(file: UploadFile = File(...)):
    """
    Validate the uploaded audio file and return the model prediction.
    """
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Only audio files are supported.",
        )

    prediction = await analyse_audio(file)

    return {"prediction": prediction}