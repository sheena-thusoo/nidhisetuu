"""Time helpers - always timezone-aware UTC."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_days(created_at: datetime, now: datetime | None = None) -> float:
    now = now or utcnow()
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (now - created_at).total_seconds() / 86400.0


def within_window(created_at: datetime, days: int, now: datetime | None = None) -> bool:
    return age_days(created_at, now) <= days


def iso_days_ago(days: int, now: datetime | None = None) -> str:
    now = now or utcnow()
    return to_iso(now - timedelta(days=days))  # type: ignore[return-value]
