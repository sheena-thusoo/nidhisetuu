"""Freshness metadata for retrieved sources: FRESH / AGING / STALE by age thresholds."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.config import get_settings
from app.utils.timeutil import age_days as compute_age_days
from app.utils.timeutil import parse_iso, to_iso

FreshnessClass = Literal["FRESH", "AGING", "STALE"]


class Freshness(BaseModel):
    retrieved_at: str
    age_days: float
    classification: FreshnessClass
    policy: dict[str, int]


def classify_freshness(
    retrieved_at: datetime | str,
    *,
    now: datetime | None = None,
    fresh_days: int | None = None,
    aging_days: int | None = None,
) -> Freshness:
    settings = get_settings()
    fresh_limit = fresh_days if fresh_days is not None else settings.freshness_fresh_days
    aging_limit = aging_days if aging_days is not None else settings.freshness_aging_days

    dt = parse_iso(retrieved_at) if isinstance(retrieved_at, str) else retrieved_at
    age = compute_age_days(dt, now)
    if age <= fresh_limit:
        classification: FreshnessClass = "FRESH"
    elif age <= aging_limit:
        classification = "AGING"
    else:
        classification = "STALE"
    return Freshness(
        retrieved_at=to_iso(dt) or "",
        age_days=round(age, 2),
        classification=classification,
        policy={"fresh_days": fresh_limit, "aging_days": aging_limit},
    )
