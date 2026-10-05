"""Application routes: create/list/get + the five-state lifecycle transitions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import AdminUser, CurrentUser, DbDep
from app.services.applications import (
    ApplicationError,
    application_audit_trail,
    create_application,
    get_application,
    list_applications,
    transition_application,
)

router = APIRouter(prefix="/applications", tags=["applications"])


class CreateApplicationRequest(BaseModel):
    scheme_id: str = Field(min_length=1, max_length=120)
    facts: dict[str, Any] = Field(default_factory=dict)


class TransitionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class DecisionRequest(TransitionRequest):
    decision: str = Field(pattern="^(approve|reject)$")


def _app_error(exc: ApplicationError) -> HTTPException:
    mapping = {
        403: status.HTTP_403_FORBIDDEN,
        404: status.HTTP_404_NOT_FOUND,
        409: status.HTTP_409_CONFLICT,
        422: status.HTTP_422_UNPROCESSABLE_ENTITY,
    }
    return HTTPException(mapping.get(exc.status_code, status.HTTP_409_CONFLICT), detail=str(exc))


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(payload: CreateApplicationRequest, user: CurrentUser, db: DbDep = None):
    try:
        application = await create_application(db, user_id=user["user_id"], scheme_id=payload.scheme_id, facts=payload.facts)
    except ApplicationError as exc:
        raise _app_error(exc) from exc
    return application


@router.get("")
async def list_mine(user: CurrentUser, db: DbDep = None, limit: int = 50):
    return {"applications": await list_applications(db, user=user, limit=limit)}


@router.get("/{application_id}")
async def get_one(application_id: str, user: CurrentUser, db: DbDep = None):
    try:
        return await get_application(db, application_id, user=user)
    except ApplicationError as exc:
        raise _app_error(exc) from exc


@router.get("/{application_id}/audit")
async def audit(application_id: str, user: CurrentUser, db: DbDep = None):
    try:
        await get_application(db, application_id, user=user)  # ownership check
    except ApplicationError as exc:
        raise _app_error(exc) from exc
    return {"events": await application_audit_trail(db, application_id)}


@router.post("/{application_id}/submit")
async def submit(application_id: str, user: CurrentUser, db: DbDep = None):
    try:
        return await transition_application(db, application_id, action="submit", user=user)
    except ApplicationError as exc:
        raise _app_error(exc) from exc


@router.post("/{application_id}/review")
async def review(application_id: str, payload: TransitionRequest, admin: AdminUser, db: DbDep = None):
    try:
        return await transition_application(db, application_id, action="review", user=admin, note=payload.note)
    except ApplicationError as exc:
        raise _app_error(exc) from exc


@router.post("/{application_id}/decision")
async def decision(application_id: str, payload: DecisionRequest, admin: AdminUser, db: DbDep = None):
    try:
        return await transition_application(db, application_id, action=payload.decision, user=admin, note=payload.note)
    except ApplicationError as exc:
        raise _app_error(exc) from exc
