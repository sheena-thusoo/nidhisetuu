"""FastAPI application factory.

Lifespan: init Mongo (real or as configured), ensure indexes, build the gateway,
warm the RAG singleton. All third-party credentials are optional by design -
missing keys mean the corresponding component runs in MOCK MODE.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.utils.logging import configure_logging, log_event


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json and settings.env != "test")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from app.db import init_database
        from app.tools.registry import build_gateway, set_gateway

        database = init_database()
        await database.ensure_indexes()
        set_gateway(build_gateway(settings))
        log_event(
            logging.getLogger(__name__),
            logging.INFO,
            "app startup complete",
            mock_mode=settings.mock_mode_flags(),
            env=settings.env,
        )
        yield
        from app.db import close_database

        await close_database()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "production" else None,
    )

    _register_middleware(app)
    _register_routers(app)
    return app


def _register_middleware(app: FastAPI) -> None:
    from app.api.middleware import register_middleware
    from app.api.ratelimit import register_rate_limiting

    register_middleware(app)
    register_rate_limiting(app)


def _register_routers(app: FastAPI) -> None:
    from app.api.routes import register_routes

    register_routes(app)


app = create_app()
