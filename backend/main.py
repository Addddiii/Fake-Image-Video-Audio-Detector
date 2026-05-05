"""
Backend for Fake Media Detection
Handles file uploads and verifies user login tokens.
"""

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import shutil
from dotenv import load_dotenv

# Import authentication functions
from auth import initialize_firebase, verify_firebase_token, get_current_user

# Import model inference functions
from inference.image import initialize_model, predict_image, get_model

# Import video inference functions
from inference.video import load_video_detector

# Import audio inference functions
from inference.audio import load_audio_detector

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Fake Media Detection API",
    description="Backend for verifying login and handling uploads",
    version="1.0.0"
)

# CORS - allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",  # In case frontend uses alternate port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", "models/image_model.pth")
VIDEO_MODEL_PATH = os.getenv("VIDEO_MODEL_PATH", "models/video_model.pth")
AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", "models/audio_model.pth")

# Global detector instances
video_detector = None
audio_detector = None


# Startup event - initialize Firebase and ML models when server starts
@app.on_event("startup")
async def startup_event():
    """Initialize Firebase Admin SDK and ML models"""
    global video_detector, audio_detector
    
    print("Starting backend...")
    initialize_firebase()
    
    # Initialize the image detection model
    print(f"Loading image model from: {IMAGE_MODEL_PATH}")
    image_model_loaded = initialize_model(IMAGE_MODEL_PATH)
    
    if image_model_loaded:
        print("✓ Image model loaded successfully!")
    else:
        print("⚠ Image model not loaded")
    
    # Initialize the video detection model
    if os.path.exists(VIDEO_MODEL_PATH):
        try:
            print(f"Loading video model from: {VIDEO_MODEL_PATH}")
            video_detector = load_video_detector(model_path=VIDEO_MODEL_PATH)
            print("✓ Video model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load video model: {e}")
    else:
        print(f"⚠ Video model not found at {VIDEO_MODEL_PATH}")
    
    # Initialize the audio detection model
    if os.path.exists(AUDIO_MODEL_PATH):
        try:
            print(f"Loading audio model from: {AUDIO_MODEL_PATH}")
            audio_detector = load_audio_detector(model_path=AUDIO_MODEL_PATH)
            print("✓ Audio model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load audio model: {e}")
    else:
        print(f"⚠ Audio model not found at {AUDIO_MODEL_PATH}")
    
    print("Backend ready!")


# ============================================
# PUBLIC ROUTES (no login required)
# ============================================

@app.get("/")
def read_root():
    """Health check"""
    return {
        "message": "Fake Media Detection Backend",
        "status": "running",
        "note": "Backend verifies login tokens - actual login happens on frontend with Firebase"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    model = get_model()
    return {
        "status": "ok",
        "models": {
            "image": {
                "loaded": model is not None,
                "path": IMAGE_MODEL_PATH
            },
            "video": {
                "loaded": video_detector is not None,
                "path": VIDEO_MODEL_PATH
            },
            "audio": {
                "loaded": audio_detector is not None,
                "path": AUDIO_MODEL_PATH
            }
        }
    }


# ============================================
# AUTHENTICATION VERIFICATION ROUTES
# ============================================

@app.post("/auth/verify")
async def verify_login(authorization: Optional[str] = Header(None)):
    """
    Verify if a user is logged in.
    """
    decoded_token = await verify_firebase_token(authorization)
    
    return {
        "valid": True,
        "message": "User is logged in",
        "user": {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False)
        }
    }


@app.get("/auth/me")
async def get_my_info(user: dict = Depends(get_current_user)):
    """
    Get information about the currently logged-in user.
    """
    return {
        "user": user,
        "message": "Successfully verified login!"
    }


# ============================================
# FILE UPLOAD (PUBLIC - no auth needed)
# ============================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload image, video, or audio file and get fake/real prediction.
    """
    
    # Determine file type
    is_image = file.content_type and file.content_type.startswith('image/')
    is_video = file.content_type and file.content_type.startswith('video/')
    is_audio = file.content_type and file.content_type.startswith('audio/')
    
    if not is_image and not is_video and not is_audio:
        raise HTTPException(
            status_code=400,
            detail="Only image, video, and audio files are supported."
        )
    
    # Save file to uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if is_image:
            model = get_model()
            
            if model is None:
                return {"error": "Image model not loaded"}
            
            prediction_result = predict_image(file_path)
            return {"prediction": prediction_result}
        
        elif is_video:
            if video_detector is None:
                return {"error": "Video model not loaded"}
            
            prediction_result = video_detector.predict(file_path)
            return {"prediction": prediction_result}
        
        elif is_audio:
            if audio_detector is None:
                return {"error": "Audio model not loaded"}
            
            prediction_result = audio_detector.predict(file_path)
            return {"prediction": prediction_result}
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)