"""Intake facts - the structured applicant profile the rule engine consumes.

`extra="allow"` matters: schemes are data-driven, so a rule may reference a field the
API does not know about yet without any code change.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LocationType = Literal["rural", "urban"]


class IntakeFacts(BaseModel):
    model_config = ConfigDict(extra="allow")

    annual_income: Decimal | None = None
    category: str | None = Field(default=None, description="SC | ST | OBC | EBC | DNT | General ...")
    location_type: LocationType | None = Field(
        default=None,
        description="rural | urban - drives rural/urban income thresholds where a scheme defines them",
    )
    state: str | None = None
    age: int | None = None
    gender: str | None = None
    course_level: str | None = Field(default=None, description="undergraduate | postgraduate | professional | diploma")
    employment_type: str | None = None
    business_type: str | None = None
    years_in_operation: Decimal | None = None
    existing_loan: bool | None = None
    loan_amount_requested: Decimal | None = None
    tenure_months: int | None = None

    def to_facts(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ExtractedField(BaseModel):
    """One field extracted by the intake agent, with extraction provenance."""

    field: str
    value: Any
    source: Literal["request", "regex", "llm"] = "request"
    confidence: float | None = None


class IntakeExtraction(BaseModel):
    facts: dict[str, Any] = Field(default_factory=dict)
    extracted: list[ExtractedField] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    provider: str = "rules"
    mock_mode: bool = False
