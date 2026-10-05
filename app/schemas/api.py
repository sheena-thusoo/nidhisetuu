"""HTTP request/response models for the agentic endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.constants import DISCLAIMER


class MatchRequest(BaseModel):
    """At least one of `query` (free text) or `facts` (structured) is required.

    Explicit `facts` always win over values extracted from `query` - the intake
    agent only fills gaps.
    """

    query: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    scheme_ids: list[str] | None = None
    top_k: int = Field(default=4, ge=1, le=10)
    allow_external_fallback: bool = True


class MatchResponse(BaseModel):
    request_id: str
    mode: str = "match"
    facts: dict[str, Any] = Field(default_factory=dict)
    unresolved_questions: list[str] = Field(default_factory=list)
    evaluations: list[dict[str, Any]] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    financials: dict[str, Any] = Field(default_factory=dict)
    partners: dict[str, Any] = Field(default_factory=dict)
    verdicts_by_scheme: dict[str, Any] = Field(default_factory=dict)
    evidence_sufficient: bool = False
    external_research: list[dict[str, Any]] = Field(default_factory=list)
    external_provider: str = "unavailable"
    external_mock_mode: bool = True
    fallback_used: bool = False
    explanation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    mock_mode: dict[str, bool] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER


class ResearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=10)


class ResearchResponse(BaseModel):
    request_id: str
    mode: str = "research"
    results: list[dict[str, Any]] = Field(default_factory=list)
    external_research: list[dict[str, Any]] = Field(default_factory=list)
    external_provider: str = "unavailable"
    external_mock_mode: bool = True
    explanation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    mock_mode: dict[str, bool] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER
