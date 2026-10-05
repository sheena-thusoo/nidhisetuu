"""Pydantic v2 schemas for the deterministic rule engine.

The rule engine consumes *data* (JSON rule specs), never hardcoded conditionals.
A RuleSpec describes one check: which field, which operator, which params.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RuleStatus = Literal["pass", "fail", "missing_data", "skipped"]
EligibilityStatus = Literal["potentially_eligible", "not_eligible", "insufficient_data"]
ThresholdDirection = Literal["lt", "lte", "gt", "gte", "eq"]


class RuleSource(BaseModel):
    """Where a rule came from - used for provenance in API responses."""

    url: str | None = None
    section: str | None = None
    page: int | None = None
    note: str | None = None


class RuleSpec(BaseModel):
    rule_id: str
    field: str
    operator: str
    description: str = ""
    # operator parameters - a rule uses only the params its operator needs
    value: Any | None = None
    values: list[Any] | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    thresholds: dict[str, Any] | None = None
    direction: ThresholdDirection | None = None
    location_field: str | None = None
    unit: str | None = None  # "INR" triggers Indian money formatting
    critical: bool = True
    source: RuleSource | None = None


class RuleResult(BaseModel):
    """Structured pass/fail per rule with actual vs threshold values."""

    rule_id: str
    description: str
    operator: str
    status: RuleStatus
    passed: bool | None
    actual: Any | None = None
    actual_display: str | None = None
    threshold: Any | None = None
    threshold_display: str | None = None
    threshold_variant: str | None = None
    comparison: str = ""
    unit: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    note: str | None = None
    critical: bool = True
    source: RuleSource | None = None


class EligibilityResult(BaseModel):
    scheme_id: str
    scheme_version: int
    status: EligibilityStatus
    passed_count: int = 0
    failed_count: int = 0
    missing_count: int = 0
    skipped_count: int = 0
    reasons: list[RuleResult] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    disclaimer: str = ""
