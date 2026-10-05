"""Shared constants (disclaimers, demo-data labels)."""

from __future__ import annotations

DISCLAIMER = (
    "Potentially eligible based on configured scheme criteria. This is not an approval and not a "
    "guarantee of credit. Final approval rests with the relevant government agency or institution."
)

ANSWERING_NOTE = "NidhiSetu is a decision-support tool. It never approves or rejects an application."

DEMO_DATA_NOTE = (
    "Placeholder demo data pending real government scheme documents. Not for official use."
)

MOCK_PROVIDER_LABEL = "mock"

EVIDENCE_STATUSES = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE",
)

FRESHNESS_CLASSES = ("FRESH", "AGING", "STALE")

JOB_STATUSES = ("QUEUED", "RUNNING", "COMPLETED", "FAILED")

APPLICATION_STATUSES = (
    "DRAFT",
    "SUBMITTED",
    "UNDER_REVIEW",
    "APPROVED",
    "REJECTED",
)
