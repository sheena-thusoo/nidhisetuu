"""Authentication: bcrypt password hashing + JWT (HS256) bearer tokens.

Registered users live in the `users` collection ({email unique, password_hash,
is_admin}). Tokens carry {sub: user_id, email, admin, iat, exp}; they never carry
application data. All failures raise AuthError, mapped to 401/403 by the API layer.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import bcrypt
import jwt

from app.config import Settings, get_settings
from app.db import Database, get_database
from app.utils.ids import new_id
from app.utils.logging import log_event, mask_email
from app.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"


class AuthError(Exception):
    def __init__(self, message: str, *, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


# ------------------------------------------------------------------ passwords


def hash_password(password: str) -> str:
    if len(password or "") < 8:
        raise AuthError("password must be at least 8 characters", status_code=422)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw((password or "").encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ------------------------------------------------------------------ tokens


def create_access_token(user_id: str, email: str, *, is_admin: bool = False, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    now = utcnow()
    payload = {
        "sub": user_id,
        "email": email,
        "admin": is_admin,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid token") from exc
    if not payload.get("sub"):
        raise AuthError("token missing subject")
    return payload


# ------------------------------------------------------------------ user store


async def register_user(
    db: Database,
    *,
    email: str,
    password: str,
    name: str | None = None,
    admin_invite_code: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    email = (email or "").strip().lower()
    if "@" not in email or len(email) > 320:
        raise AuthError("a valid email is required", status_code=422)
    is_admin = bool(
        admin_invite_code
        and settings.admin_api_key
        and admin_invite_code == settings.admin_api_key
    )
    if admin_invite_code and not is_admin:
        raise AuthError("invalid admin invite code", status_code=403)

    existing = await db.users.find_one({"email": email})
    if existing:
        raise AuthError("email already registered", status_code=409)

    user = {
        "user_id": new_id(),
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "is_admin": is_admin,
        "created_at": utcnow().isoformat(),
    }
    await db.users.insert_one(dict(user))
    log_event(logger, logging.INFO, "user registered", email=mask_email(email), admin=is_admin)
    # return a copy without secret or Mongo internals
    return {k: v for k, v in user.items() if k not in ("password_hash", "_id")}


async def authenticate(db: Database, *, email: str, password: str) -> dict[str, Any]:
    email = (email or "").strip().lower()
    user = await db.users.find_one({"email": email})
    if user is None or not verify_password(password, user.get("password_hash", "")):
        raise AuthError("invalid email or password")
    return {k: v for k, v in user.items() if k not in ("password_hash", "_id")}


async def user_from_token(db: Database, token: str, settings: Settings | None = None) -> dict[str, Any]:
    payload = decode_token(token, settings)
    user = await db.users.find_one({"user_id": payload["sub"]})
    if user is None:
        raise AuthError("user no longer exists")
    return {k: v for k, v in user.items() if k not in ("password_hash", "_id")}
