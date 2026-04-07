"""
Authentication module for Firebase integration.
Backend ONLY verifies login tokens - Firebase handles the actual login.
"""

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Header
from typing import Optional
import os

# Flag to check if Firebase is properly configured
firebase_initialized = False


def initialize_firebase():
    """
    Set up Firebase Admin SDK.
    This runs once when the server starts.
    
    Backend uses this to VERIFY tokens that Firebase creates on the frontend.
    We're NOT doing the actual login here - that's handled by Firebase on frontend.
    """
    global firebase_initialized
    
    # Don't initialize twice - Firebase doesn't like that!
    if firebase_initialized:
        return
    
    # Path to the Firebase service account JSON file
    # Check multiple possible locations
    possible_paths = [
        os.getenv('FIREBASE_CREDENTIALS_PATH'),  # From environment variable
        'firebase-credentials.json',  # Same directory as auth.py
        'backend/firebase-credentials.json',  # From project root
        os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')  # Absolute path
    ]
    
    cred_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            cred_path = path
            break
    
    if not cred_path:
        cred_path = 'backend/firebase-credentials.json'  # Default for error message
    
    try:
        # Check if the credentials file exists
        if os.path.exists(cred_path):
            # Initialize Firebase Admin with the service account
            if not firebase_admin._apps: 
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

            firebase_initialized = True
            print("Firebase Admin initialized - can verify login tokens!")
        else:
            # No credentials file - backend can't verify tokens
            print(f"⚠️  Warning: Firebase credentials not found")
            print(f"   Looked in: {', '.join([p for p in possible_paths if p])}")
            print("   Download from: Firebase Console -> Project Settings -> Service Accounts")
    except Exception as e:
        # Something went wrong during initialization
        print(f"Error initializing Firebase: {str(e)}")
        print("   Backend won't be able to verify login tokens.")


async def verify_firebase_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verify that a user is actually logged in by checking their Firebase token.
    
    Flow:
    1. User logs in on FRONTEND using Firebase
    2. Firebase gives them a token
    3. Frontend sends token to backend in Authorization header
    4. Backend (this function) verifies it's a real token
    
    Args:
        authorization: The Authorization header from request (format: "Bearer <token>")
        
    Returns:
        dict: User info from the token (uid, email, etc.)
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    
    # If Firebase isn't set up, we can't verify tokens
    if not firebase_initialized:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Token verification not available. Backend needs Firebase credentials."
        )
    
    # Check if the Authorization header exists
    if not authorization:
        raise HTTPException(
            status_code=401,  # Unauthorized
            detail="No authorization token provided. Please log in."
        )
    
    # Extract the token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        
        # Make sure it's a Bearer token
        if scheme.lower() != 'bearer':
            raise HTTPException(
                status_code=401,
                detail="Invalid auth format. Use 'Bearer <token>'"
            )
    except ValueError:
        # Header wasn't formatted correctly
        raise HTTPException(
            status_code=401,
            detail="Invalid auth header. Expected 'Bearer <token>'"
        )
    
    # Verify the token with Firebase
    try:
        # Firebase checks if the token is valid and not expired
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except auth.InvalidIdTokenError:
        # Token is fake or expired
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again."
        )
    except Exception as e:
        # Some other error
        raise HTTPException(
            status_code=401,
            detail=f"Token verification failed: {str(e)}"
        )


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Get info about the currently logged-in user.
    
    This verifies the token and returns clean user info.
    Use this in routes where you need to know who's making the request.
    
    Args:
        authorization: The Authorization header
        
    Returns:
        dict: User information (uid, email, name, etc.)
    """
    
    # First verify the token
    decoded_token = await verify_firebase_token(authorization)
    
    # Extract user info from the token
    user_info = {
        "uid": decoded_token.get("uid"),  # Unique user ID
        "email": decoded_token.get("email"),  # User's email
        "email_verified": decoded_token.get("email_verified", False),  # Email verified?
        "name": decoded_token.get("name"),  # Display name (if they set one)
        "picture": decoded_token.get("picture"),  # Profile pic (if using Google/Facebook)
    }
    
    return user_info