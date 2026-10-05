"""ID helpers kept separate so logging can import them without cycles."""

from __future__ import annotations

import uuid


def new_request_id() -> str:
    return uuid.uuid4().hex


def new_id() -> str:
    return uuid.uuid4().hex


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]
