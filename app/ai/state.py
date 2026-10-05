"""LangGraph state for both flows (match / research)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.ai.explanation import ExplanationResult
from app.schemas.intake import IntakeExtraction
from app.schemas.partners import PartnerMatch
from app.schemas.rag import EvidenceClaim, EvidenceVerdict, RetrievedChunk
from app.schemas.research import ExternalResearchItem
from app.services.eligibility import EvaluatedFinancials, SchemeEvaluation

FlowMode = Literal["match", "research"]


class GraphState(TypedDict, total=False):
    # inputs
    mode: FlowMode
    request_id: str
    raw_query: str
    provided_facts: dict[str, Any]
    scheme_ids: list[str] | None
    top_k: int
    allow_external_fallback: bool

    # intake
    extraction: IntakeExtraction
    facts: dict[str, Any]

    # deterministic eligibility + financials
    evaluations: list[SchemeEvaluation]
    candidate_ids: list[str]
    financials: dict[str, EvaluatedFinancials]
    partners: dict[str, list[PartnerMatch]]

    # retrieval + verification
    chunks_by_scheme: dict[str, list[RetrievedChunk]]
    chunks_global: list[RetrievedChunk]
    claims_by_scheme: dict[str, list[EvidenceClaim]]
    verdicts_by_scheme: dict[str, list[EvidenceVerdict]]
    research_verdicts: list[dict[str, Any]]
    evidence_sufficient: bool
    scheme_evidence_ok: dict[str, bool]

    # external research
    external_research: list[ExternalResearchItem]
    external_provider: str
    external_mock_mode: bool
    fallback_used: bool

    # output
    explanation: ExplanationResult
    providers: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    tool_trace: list[dict[str, Any]]


def add_warning(state: GraphState, message: str) -> list[str]:
    warnings = list(state.get("warnings", []))
    warnings.append(message)
    return warnings


def add_trace(state: GraphState, tool_result) -> list[dict[str, Any]]:
    trace = list(state.get("tool_trace", []))
    trace.append(tool_result.trace_entry())
    return trace
