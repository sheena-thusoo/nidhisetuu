"""Application lifecycle: DRAFT -> SUBMITTED -> UNDER_REVIEW -> APPROVED | REJECTED.

Five states, explicit transition table, ownership checks, and an append-only audit
trail in the `audit_events` collection. Nothing about eligibility is decided here:
the applicant's snapshot stores the deterministic evaluation computed at creation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import Database
from app.rules.loader import get_rule_repository
from app.services.eligibility import match_schemes
from app.utils.ids import new_id
from app.utils.logging import log_event
from app.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

DRAFT = "DRAFT"
SUBMITTED = "SUBMITTED"
UNDER_REVIEW = "UNDER_REVIEW"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
APPLICATION_STATUSES = (DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED)

# action -> (allowed from-states, required role)
TRANSITIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "submit": ((DRAFT,), "owner"),
    "review": ((SUBMITTED,), "admin"),
    "approve": ((UNDER_REVIEW,), "admin"),
    "reject": ((UNDER_REVIEW,), "admin"),
}


class ApplicationError(Exception):
    def __init__(self, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


async def _audit(db: Database, *, application_id: str, actor_id: str, action: str, from_status: str, to_status: str, note: str | None = None) -> None:
    await db.audit_events.insert_one(
        {
            "event_id": new_id(),
            "entity": "application",
            "entity_id": application_id,
            "actor_id": actor_id,
            "action": action,
            "from_status": from_status,
            "to_status": to_status,
            "note": note,
            "created_at": utcnow().isoformat(),
        }
    )


async def create_application(
    db: Database,
    *,
    user_id: str,
    scheme_id: str,
    facts: dict[str, Any],
) -> dict[str, Any]:
    scheme = get_rule_repository().get(scheme_id)
    if scheme is None:
        raise ApplicationError(f"unknown scheme_id {scheme_id!r}", status_code=422)

    # deterministic eligibility snapshot at creation time (never recomputed by an LLM)
    evaluation = match_schemes(get_rule_repository(), facts, scheme_ids=[scheme_id])[0]

    now = utcnow().isoformat()
    application = {
        "application_id": new_id(),
        "user_id": user_id,
        "scheme_id": scheme_id,
        "scheme_name": scheme.name,
        "scheme_version": scheme.version,
        "status": DRAFT,
        "facts": facts,
        "eligibility_snapshot": {
            "status": evaluation.eligibility.status,
            "missing_fields": evaluation.eligibility.missing_fields,
            "computed_by": "app/services/eligibility.py",
        },
        "created_at": now,
        "updated_at": now,
    }
    await db.applications.insert_one(application)
    await _audit(db, application_id=application["application_id"], actor_id=user_id, action="create", from_status="-", to_status=DRAFT)
    return {k: v for k, v in application.items() if k != "_id"}


async def get_application(db: Database, application_id: str, *, user: dict[str, Any]) -> dict[str, Any]:
    application = await db.applications.find_one({"application_id": application_id})
    if application is None:
        raise ApplicationError("application not found", status_code=404)
    if application["user_id"] != user["user_id"] and not user.get("is_admin"):
        # do not reveal existence to non-owners
        raise ApplicationError("application not found", status_code=404)
    return {k: v for k, v in application.items() if k != "_id"}


async def list_applications(db: Database, *, user: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
    query = {} if user.get("is_admin") else {"user_id": user["user_id"]}
    cursor = db.applications.find(query, {"_id": 0}).sort("created_at", -1).limit(max(1, min(limit, 100)))
    return [doc async for doc in cursor]


async def transition_application(
    db: Database,
    application_id: str,
    *,
    action: str,
    user: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    if action not in TRANSITIONS:
        raise ApplicationError(f"unknown action {action!r}", status_code=422)
    allowed_from, required_role = TRANSITIONS[action]

    application = await db.applications.find_one({"application_id": application_id})
    if application is None:
        raise ApplicationError("application not found", status_code=404)

    if required_role == "owner":
        if application["user_id"] != user["user_id"]:
            raise ApplicationError("only the owner can do that", status_code=403)
    else:
        if not user.get("is_admin"):
            raise ApplicationError("admin role required", status_code=403)

    current = application["status"]
    if current not in allowed_from:
        raise ApplicationError(
            f"cannot {action} an application in status {current} (allowed from: {', '.join(allowed_from)})",
            status_code=409,
        )

    to_status = {"submit": SUBMITTED, "review": UNDER_REVIEW, "approve": APPROVED, "reject": REJECTED}[action]
    await db.applications.update_one(
        {"application_id": application_id, "status": current},  # optimistic guard
        {"$set": {"status": to_status, "updated_at": utcnow().isoformat()}},
    )
    await _audit(
        db, application_id=application_id, actor_id=user["user_id"],
        action=action, from_status=current, to_status=to_status, note=note,
    )
    log_event(
        logger, logging.INFO, "application transition",
        application_id=application_id, action=action, from_status=current, to_status=to_status,
    )
    return await get_application(db, application_id, user=user)


async def application_audit_trail(db: Database, application_id: str) -> list[dict[str, Any]]:
    cursor = (
        db.audit_events.find({"entity": "application", "entity_id": application_id}, {"_id": 0})
        .sort("created_at", 1)
    )
    return [doc async for doc in cursor]
