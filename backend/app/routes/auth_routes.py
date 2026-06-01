from typing import Optional

from fastapi import APIRouter, Depends, Header

from app.auth import get_current_user, verify_firebase_token


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/verify")
async def verify_login(authorization: Optional[str] = Header(None)):
    decoded_token = await verify_firebase_token(authorization)

    return {
        "valid": True,
        "user": {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
        },
    }


@router.get("/me")
async def get_my_info(user: dict = Depends(get_current_user)):
    return {"user": user}