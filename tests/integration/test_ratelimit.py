"""Phase 7: rate-limiting wiring.

The suite runs with RATE_LIMIT_ENABLED=false (main app limiter disabled), so the
429 behaviour is verified on a standalone mini-app with the same limiter factory
but an explicit low limit. This proves the wiring, not the production limits.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _mini_app(limit: str = "2/minute") -> FastAPI:
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address, enabled=True)

    @app.get("/limited")
    @limiter.limit(limit)
    async def limited(request: Request):
        return {"ok": True}

    from slowapi import _rate_limit_exceeded_handler

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return app


def test_rate_limit_returns_429_after_budget_exhausted():
    client = TestClient(_mini_app("2/minute"))
    assert client.get("/limited").status_code == 200
    assert client.get("/limited").status_code == 200
    third = client.get("/limited")
    assert third.status_code == 429
    assert "limit" in third.json()["error"].lower() or "rate" in third.json()["error"].lower()


def test_main_app_limiter_is_disabled_in_test_env():
    from fastapi.testclient import TestClient as TC

    from app.main import create_app

    client = TC(create_app())
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert client.app.state.limiter.enabled is False


def test_production_settings_define_the_documented_limits():
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    assert settings.rate_limit_match == "30/minute"
    assert settings.rate_limit_research == "15/minute"
    assert settings.rate_limit_auth == "20/minute"
