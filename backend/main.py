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
from model_inference import initialize_model, predict_image, get_model

# Import video inference functions
from video_inference import load_video_detector

# Import audio inference functions
from audio_inference import load_audio_detector

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
IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", "deepfake_model.pth")
VIDEO_MODEL_PATH = os.getenv("VIDEO_MODEL_PATH", "deepfake_video_model_best.pth")
AUDIO_MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", "best_audio_model.pth")

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
        },
        "endpoints": {
            "public": {
                "/": "Health check",
                "/upload": "Upload image, video, or audio and get fake/real prediction"
            },
            "auth_verification": {
                "/auth/verify": "Verify if user is logged in",
                "/auth/me": "Get current user info (requires login token)"
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
    
    Frontend: User logs in with Firebase -> gets token
    Frontend: Sends token to this endpoint
    Backend: Verifies token is real -> returns user info
    
    Headers:
        Authorization: Bearer <firebase-token>
    
    Returns:
        User info if logged in, error if not
    """
    try:
        # Verify the token with Firebase
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
    except HTTPException as e:
        # Token is invalid
        raise e


@app.get("/auth/me")
async def get_my_info(user: dict = Depends(get_current_user)):
    """
    Get information about the currently logged-in user.
    
    This endpoint requires a valid login token.
    Frontend sends the Firebase token, backend verifies it and returns user info.
    
    Headers:
        Authorization: Bearer <firebase-token>
    
    Returns:
        Current user's profile information
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
    
    Args:
        file: Image file (jpg, png, webp), video file (mp4, avi, mov), or audio file (wav, mp3, flac)
    
    Returns:
        JSON with prediction results including:
        - prediction: 'fake' or 'real'
        - confidence: confidence percentage (0-100)
        - probabilities: breakdown of fake and real probabilities
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
        
        # Process based on file type
        if is_image:
            # Get image model instance
            model = get_model()
            
            if model is None:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "File uploaded successfully",
                    "error": "Image model not loaded"
                }
            
            # Make prediction on the uploaded image
            try:
                prediction_result = predict_image(file_path)
                
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Image analyzed successfully",
                    "prediction": prediction_result
                }
            
            except Exception as pred_error:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Image uploaded but prediction failed",
                    "error": str(pred_error)
                }
        
        elif is_video:
            # Check if video model is loaded
            if video_detector is None:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "File uploaded successfully",
                    "error": "Video model not loaded"
                }
            
            # Make prediction on the uploaded video
            try:
                prediction_result = video_detector.predict(file_path)
                
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Video analyzed successfully",
                    "prediction": prediction_result
                }
            
            except Exception as pred_error:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Video uploaded but prediction failed",
                    "error": str(pred_error)
                }
        
        elif is_audio:
            # Check if audio model is loaded
            if audio_detector is None:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "File uploaded successfully",
                    "error": "Audio model not loaded"
                }
            
            # Make prediction on the uploaded audio
            try:
                prediction_result = audio_detector.predict(file_path)
                
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Audio analyzed successfully",
                    "prediction": prediction_result
                }
            
            except Exception as pred_error:
                return {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "message": "Audio uploaded but prediction failed",
                    "error": str(pred_error)
                }
    
    except Exception as e:
        # Clean up file if it was created
        if os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
    
    finally:
        # Optional: Clean up uploaded file after processing
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)