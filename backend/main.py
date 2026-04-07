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


# Startup event - initialize Firebase when server starts
@app.on_event("startup")
async def startup_event():
    """Initialize Firebase Admin SDK for token verification"""
    print("Starting backend...")
    initialize_firebase()
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
    return {
        "status": "ok",
        "endpoints": {
            "public": {
                "/": "Health check",
                "/upload": "Upload files (no auth needed)"
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
    Upload file to backend (currently public).
    Later this can be protected using authentication.
    """

    # Save file to uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File uploaded successfully"
    }


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)