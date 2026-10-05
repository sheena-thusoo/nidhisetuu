"""Pydantic input/output schemas for every gateway tool (documented in /api/tools)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.ai.explanation import ExplanationResult
from app.schemas.intake import IntakeExtraction
from app.schemas.partners import PartnerMatch
from app.schemas.rag import EvidenceClaim, EvidenceVerdict, RetrievedChunk
from app.schemas.research import ExternalResearchItem
from app.services.eligibility import EvaluatedFinancials, SchemeEvaluation


class ExtractIntakeInput(BaseModel):
    text: str | None = Field(default=None, description="Free-text applicant description")
    provided: dict[str, Any] = Field(default_factory=dict, description="Structured values already supplied")


class ExtractIntakeOutput(BaseModel):
    extraction: IntakeExtraction


class CheckEligibilityInput(BaseModel):
    facts: dict[str, Any]
    scheme_ids: list[str] | None = None


class CheckEligibilityOutput(BaseModel):
    evaluations: list[SchemeEvaluation]


class CalculateFinancialsInput(BaseModel):
    scheme_id: str
    facts: dict[str, Any]


class CalculateFinancialsOutput(BaseModel):
    scheme_id: str
    financials: EvaluatedFinancials


class MatchPartnersInput(BaseModel):
    scheme_ids: list[str]
    state: str | None = None


class MatchPartnersOutput(BaseModel):
    matches: dict[str, list[PartnerMatch]]


class RagSearchInput(BaseModel):
    query: str
    scheme_id: str | None = None
    top_k: int = 4


class RagSearchOutput(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    provider: str = "memory"
    mock_mode: bool = True


class VerifyClaimsInput(BaseModel):
    claims: list[EvidenceClaim]
    chunks: list[RetrievedChunk]
    scheme_id: str | None = None


class VerifyClaimsOutput(BaseModel):
    verdicts: list[EvidenceVerdict]


class ExternalResearchInput(BaseModel):
    query: str
    max_results: int = 2


class ExternalResearchOutput(BaseModel):
    items: list[ExternalResearchItem] = Field(default_factory=list)
    provider: str = "tavily"
    mock_mode: bool = True


class GenerateExplanationInput(BaseModel):
    payload: dict[str, Any]


class GenerateExplanationOutput(BaseModel):
    explanation: ExplanationResult
