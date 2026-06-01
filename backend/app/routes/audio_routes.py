from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.audio_service import analyse_audio


router = APIRouter(prefix="/predict", tags=["Audio"])


@router.post("/audio")
async def predict_audio_route(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are supported.")

    result = await analyse_audio(file)
    return {"prediction": result}