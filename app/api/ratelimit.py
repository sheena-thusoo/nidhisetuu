"""Rate limiting (slowapi) - enabled in production, no-op when disabled.

Limits (from settings): /schemes/match 30/min, /research 15/min, /auth 20/min.
Keyed by client IP. When `RATE_LIMIT_ENABLED=false` (tests, local dev) the limiter
is constructed disabled and every decorator is a no-op.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings

_settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    enabled=_settings.rate_limit_enabled,
    default_limits=[],  # only explicit endpoints are limited
)


def register_rate_limiting(app) -> None:
    """Attach the limiter + 429 handler to the app (idempotent per app instance)."""
    from slowapi.middleware import SlowAPIMiddleware

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    from slowapi import _rate_limit_exceeded_handler

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
