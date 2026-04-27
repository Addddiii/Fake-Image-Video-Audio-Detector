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

# Model path - set this to where your trained model is located
# You can set via environment variable: MODEL_PATH=/path/to/your/deepfake_model.pth
MODEL_PATH = os.getenv("MODEL_PATH", "deepfake_model.pth")


# Startup event - initialize Firebase and ML model when server starts
@app.on_event("startup")
async def startup_event():
    """Initialize Firebase Admin SDK and ML model"""
    print("Starting backend...")
    initialize_firebase()
    
    # Initialize the ML model
    print(f"Loading ML model from: {MODEL_PATH}")
    model_loaded = initialize_model(MODEL_PATH)
    
    if model_loaded:
        print("✓ ML model loaded successfully!")
    else:
        print("⚠ ML model not loaded - place your trained .pth file at:")
        print(f"  {os.path.abspath(MODEL_PATH)}")
    
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
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else "Model not loaded",
        "endpoints": {
            "public": {
                "/": "Health check",
                "/upload": "Upload image and get fake/real prediction"
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
    Upload image file and get fake/real prediction.
    
    Args:
        file: Image file (jpg, png, webp)
    
    Returns:
        JSON with prediction results including:
        - prediction: 'fake' or 'real'
        - confidence: confidence percentage (0-100)
        - probabilities: breakdown of fake and real probabilities
    """
    
    # Validate file type - only process images for now
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported. Video and audio detection coming soon."
        )
    
    # Save file to uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get model instance
        model = get_model()
        
        if model is None:
            # Model not loaded - return upload success without prediction
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "message": "File uploaded successfully",
                "error": "Model not loaded - prediction unavailable. Please add your trained model file."
            }
        
        # Make prediction on the uploaded image
        try:
            prediction_result = predict_image(file_path)
            
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "message": "File uploaded and analyzed successfully",
                "prediction": prediction_result
            }
        
        except Exception as pred_error:
            # Prediction failed but file was uploaded
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "message": "File uploaded but prediction failed",
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
        # Uncomment these lines if you don't want to keep uploaded files
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)