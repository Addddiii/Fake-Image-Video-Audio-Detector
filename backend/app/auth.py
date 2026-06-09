"""
Firebase authentication helpers.
"""

import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Header, HTTPException

firebase_initialised = False


def get_firebase_credentials_path() -> Optional[Path]:
    """
    Find the Firebase credentials file from environment variables or common paths.
    """
    possible_paths = [
        os.getenv("FIREBASE_CREDENTIALS_PATH"),
        "firebase-credentials.json",
        "backend/firebase-credentials.json",
        Path(__file__).resolve().parents[1] / "firebase-credentials.json",
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            return Path(path)

    return None


def initialise_firebase() -> None:
    """
    Initialise Firebase Admin SDK once using the available credentials file.
    """
    global firebase_initialised

    if firebase_initialised:
        return

    credential_path = get_firebase_credentials_path()

    if credential_path is None:
        return

    try:
        if not firebase_admin._apps:
            credential = credentials.Certificate(str(credential_path))
            firebase_admin.initialize_app(credential)

        firebase_initialised = True

    except Exception:
        firebase_initialised = False


initialize_firebase = initialise_firebase


async def verify_firebase_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """
    Verify a Firebase ID token from the Authorization header.
    """
    if not firebase_initialised:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured.",
        )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization token missing.",
        )

    try:
        scheme, token = authorization.split()

        if scheme.lower() != "bearer":
            raise ValueError

    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format.",
        )

    try:
        return auth.verify_id_token(token)

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Token verification failed.",
        )


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> dict:
    """
    Return the authenticated user's Firebase profile details.
    """
    decoded_token = await verify_firebase_token(authorization)

    return {
        "uid": decoded_token.get("uid"),
        "email": decoded_token.get("email"),
        "email_verified": decoded_token.get("email_verified", False),
        "name": decoded_token.get("name"),
        "picture": decoded_token.get("picture"),
    }