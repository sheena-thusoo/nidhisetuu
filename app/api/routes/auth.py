"""Auth routes: register, login, me. Bearer JWT in the Authorization header.

NOTE: no `from __future__ import annotations` - see app/api/routes/agent.py for why
(slowapi wrapping + postponed annotations breaks FastAPI's type resolution).
"""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import CurrentUser, DbDep, SettingsDep
from app.api.ratelimit import limiter
from app.services import auth as auth_service
from app.services.auth import AuthError

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=120)
    admin_invite_code: str | None = Field(default=None, description="ADMIN_API_KEY to create an admin user")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _auth_error(exc: AuthError) -> HTTPException:
    code = status.HTTP_401_UNAUTHORIZED
    if exc.status_code == 403:
        code = status.HTTP_403_FORBIDDEN
    elif exc.status_code == 409:
        code = status.HTTP_409_CONFLICT
    elif exc.status_code == 422:
        code = status.HTTP_422_UNPROCESSABLE_ENTITY
    return HTTPException(code, detail=str(exc))


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def register(request: Request, payload: RegisterRequest, db: DbDep = None, settings: SettingsDep = None):
    try:
        user = await auth_service.register_user(
            db,
            email=payload.email,
            password=payload.password,
            name=payload.name,
            admin_invite_code=payload.admin_invite_code,
            settings=settings,
        )
    except AuthError as exc:
        raise _auth_error(exc) from exc
    token = auth_service.create_access_token(
        user["user_id"], user["email"], is_admin=user.get("is_admin", False), settings=settings
    )
    return TokenResponse(access_token=token, user=user)


@router.post("/login", response_model=TokenResponse)
@limiter.limit("20/minute")
async def login(request: Request, payload: LoginRequest, db: DbDep = None, settings: SettingsDep = None):
    try:
        user = await auth_service.authenticate(db, email=payload.email, password=payload.password)
    except AuthError as exc:
        raise _auth_error(exc) from exc
    token = auth_service.create_access_token(
        user["user_id"], user["email"], is_admin=user.get("is_admin", False), settings=settings
    )
    return TokenResponse(access_token=token, user=user)


@router.get("/me")
async def me(user: CurrentUser):
    return {"user": user}
