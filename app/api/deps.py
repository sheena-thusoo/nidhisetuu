"""FastAPI dependencies: Mongo handle, settings, admin-key gate, JWT current-user."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.db import Database, get_database
from app.services import auth as auth_service
from app.services.auth import AuthError

_bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Database:
    return get_database()


def require_admin_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_key: Annotated[str | None, Header()] = None,
) -> str:
    """Admin endpoints require either the ADMIN_API_KEY header or an admin JWT (Phase 6)."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin API not configured (set ADMIN_API_KEY)",
        )
    if not x_admin_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Admin-Key header required")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid admin key")
    return x_admin_key


DbDep = Annotated[Database, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AdminDep = Annotated[str, Depends(require_admin_key)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    db: DbDep,
    settings: SettingsDep,
) -> dict:
    """Decode the Bearer JWT and load the user. 401 on anything unusual."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authorization: Bearer <token> required")
    try:
        return await auth_service.user_from_token(db, credentials.credentials, settings)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


CurrentUser = Annotated[dict, Depends(get_current_user)]


async def require_admin_user(user: CurrentUser) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin role required")
    return user


AdminUser = Annotated[dict, Depends(require_admin_user)]
