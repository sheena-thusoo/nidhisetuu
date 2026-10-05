"""System routes: health + tool catalog."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import SettingsDep
from app.config import get_settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(settings: SettingsDep = None):
    """Liveness + configured-mode report (which components are in MOCK MODE)."""
    settings = settings or get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.env,
        "mock_mode": settings.mock_mode_flags(),
    }


@router.get("/api/tools")
async def tools(settings: SettingsDep = None):
    """The tool catalog the agent layer uses (deterministic tools flagged)."""
    from app.tools.registry import get_gateway

    gateway = get_gateway()
    return {
        "tools": gateway.descriptors(),
        "mock_mode": (settings or get_settings()).mock_mode_flags(),
    }
