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
    global firebase_initialised

    if firebase_initialised:
        return

    cred_path = get_firebase_credentials_path()

    if not cred_path:
        return

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred)

        firebase_initialised = True

    except Exception:
        firebase_initialised = False


initialize_firebase = initialise_firebase


async def verify_firebase_token(
    authorisation: Optional[str] = Header(None, alias="Authorisation"),
) -> dict:
    if not firebase_initialised:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured.",
        )

    if not authorisation:
        raise HTTPException(
            status_code=401,
            detail="Authorisation token missing.",
        )

    try:
        scheme, token = authorisation.split()

        if scheme.lower() != "bearer":
            raise ValueError

    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authorisation header format.",
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
    authorisation: Optional[str] = Header(None, alias="Authorisation"),
) -> dict:
    decoded_token = await verify_firebase_token(authorisation)

    return {
        "uid": decoded_token.get("uid"),
        "email": decoded_token.get("email"),
        "email_verified": decoded_token.get("email_verified", False),
        "name": decoded_token.get("name"),
        "picture": decoded_token.get("picture"),
    }