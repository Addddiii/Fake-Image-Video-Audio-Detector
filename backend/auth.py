"""
Firebase Authentication Module

This module initializes Firebase Admin SDK and verifies user authentication tokens.
The frontend handles login. The backend only verifies tokens.
"""

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Header
from typing import Optional
import os

# Tracks whether Firebase has been initialized
firebase_initialized = False


def initialize_firebase():
    """
    Initialize Firebase Admin SDK.

    This runs once when the backend starts.
    It allows the backend to verify tokens issued by Firebase on the frontend.
    """
    global firebase_initialized

    # Prevent multiple initializations
    if firebase_initialized:
        return

    # Possible locations for Firebase credentials file
    possible_paths = [
        os.getenv('FIREBASE_CREDENTIALS_PATH'),
        'firebase-credentials.json',
        'backend/firebase-credentials.json',
        os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')
    ]

    cred_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            cred_path = path
            break

    if not cred_path:
        cred_path = 'backend/firebase-credentials.json'

    try:
        if os.path.exists(cred_path):
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

            firebase_initialized = True
            print("Firebase initialized successfully")
        else:
            print("Firebase credentials not found")
            print("Checked paths:", possible_paths)

    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        print("Token verification will not work")


async def verify_firebase_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verify a Firebase authentication token.

    Args:
        authorization: Authorization header ("Bearer <token>")

    Returns:
        Decoded token containing user information

    Raises:
        HTTPException if token is missing or invalid
    """

    if not firebase_initialized:
        raise HTTPException(
            status_code=503,
            detail="Firebase not initialized. Cannot verify tokens."
        )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization token missing"
        )

    try:
        scheme, token = authorization.split()

        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid token format. Use 'Bearer <token>'"
            )

    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format"
        )

    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token verification failed: {str(e)}"
        )


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Retrieve information about the currently authenticated user.

    Args:
        authorization: Authorization header

    Returns:
        Dictionary containing user details
    """

    decoded_token = await verify_firebase_token(authorization)

    user_info = {
        "uid": decoded_token.get("uid"),
        "email": decoded_token.get("email"),
        "email_verified": decoded_token.get("email_verified", False),
        "name": decoded_token.get("name"),
        "picture": decoded_token.get("picture"),
    }

    return user_info