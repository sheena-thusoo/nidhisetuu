"""Freshness classification: FRESH / AGING / STALE with exact boundary behaviour."""

from __future__ import annotations

from datetime import timedelta

from app.services.freshness import classify_freshness
from app.utils.timeutil import utcnow


def days_ago(days: int, now=None):
    return (now or utcnow()) - timedelta(days=days)


def test_fresh_now():
    now = utcnow()
    result = classify_freshness(days_ago(0, now), now=now)
    assert result.classification == "FRESH"
    assert result.age_days == 0.0


def test_fresh_boundary_exactly_180_days():
    now = utcnow()
    assert classify_freshness(days_ago(180, now), now=now).classification == "FRESH"
    assert classify_freshness(days_ago(181, now), now=now).classification == "AGING"


def test_aging_range():
    now = utcnow()
    assert classify_freshness(days_ago(200, now), now=now).classification == "AGING"


def test_stale_boundary_exactly_365_days():
    now = utcnow()
    assert classify_freshness(days_ago(365, now), now=now).classification == "AGING"
    assert classify_freshness(days_ago(366, now), now=now).classification == "STALE"


def test_policy_is_reported():
    now = utcnow()
    result = classify_freshness(days_ago(10, now), now=now)
    assert result.policy == {"fresh_days": 180, "aging_days": 365}


def test_accepts_iso_string_input():
    iso = days_ago(3).isoformat()
    result = classify_freshness(iso)
    assert result.classification == "FRESH"
    assert result.retrieved_at == iso


def test_custom_thresholds():
    assert classify_freshness(days_ago(40), fresh_days=30, aging_days=90).classification == "AGING"
