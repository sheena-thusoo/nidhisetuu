"""Route registration for the FastAPI app."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import admin, agent, applications, auth, system


def register_routes(app: FastAPI) -> None:
    app.include_router(system.router)
    app.include_router(admin.router)
    app.include_router(agent.router)
    app.include_router(auth.router)
    app.include_router(applications.router)


__all__ = ["register_routes"]
