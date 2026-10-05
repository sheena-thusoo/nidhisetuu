"""Tool registry - the only place tool handlers are bound to the gateway.

Deterministic tools (never_llm=True) wrap app/services and app/rules. Agentic tools wrap
app/ai and app/rag. Nothing calls these handlers directly; the LangGraph nodes always go
through `ToolGateway.call` so validation/timeout/logging/trace are uniform.
"""

from __future__ import annotations

import logging

from app.ai.explanation import generate_explanation
from app.ai.intake import run_intake
from app.config import Settings, get_settings
from app.rag.evidence import verify_claims
from app.rag.retrieval import get_rag_service
from app.rules.loader import get_rule_repository
from app.services.eligibility import build_financials, match_schemes
from app.services.partners import get_partner_service
from app.tools.gateway import ToolGateway
from app.tools.schemas import (
    CalculateFinancialsInput,
    CalculateFinancialsOutput,
    CheckEligibilityInput,
    CheckEligibilityOutput,
    ExternalResearchInput,
    ExternalResearchOutput,
    ExtractIntakeInput,
    ExtractIntakeOutput,
    GenerateExplanationInput,
    GenerateExplanationOutput,
    MatchPartnersInput,
    MatchPartnersOutput,
    RagSearchInput,
    RagSearchOutput,
    VerifyClaimsInput,
    VerifyClaimsOutput,
)

logger = logging.getLogger(__name__)


def register_core_tools(gateway: ToolGateway, settings: Settings) -> None:
    async def extract_intake_handler(payload: ExtractIntakeInput) -> ExtractIntakeOutput:
        extraction = await run_intake(payload.text, payload.provided)
        return ExtractIntakeOutput(extraction=extraction)

    async def check_eligibility_handler(payload: CheckEligibilityInput) -> CheckEligibilityOutput:
        repo = get_rule_repository()
        evaluations = match_schemes(
            repo, payload.facts, scheme_ids=payload.scheme_ids, include_financials=False
        )
        return CheckEligibilityOutput(evaluations=evaluations)

    async def calculate_financials_handler(payload: CalculateFinancialsInput) -> CalculateFinancialsOutput:
        scheme = get_rule_repository().get(payload.scheme_id)
        if scheme is None:
            raise ValueError(f"unknown scheme_id {payload.scheme_id!r}")
        financials = build_financials(scheme, payload.facts)
        return CalculateFinancialsOutput(scheme_id=payload.scheme_id, financials=financials)

    async def match_partners_handler(payload: MatchPartnersInput) -> MatchPartnersOutput:
        matches = get_partner_service().match_many(payload.scheme_ids, payload.state)
        return MatchPartnersOutput(matches=matches)

    gateway.register(
        name="extract_intake",
        description="Extract structured applicant facts (income, category, rural/urban, age, course) from free text.",
        input_model=ExtractIntakeInput,
        output_model=ExtractIntakeOutput,
        handler=extract_intake_handler,
        mock_mode=not settings.groq_enabled,
    )
    gateway.register(
        name="check_eligibility",
        description="DETERMINISTIC. Evaluate scheme eligibility rules and return per-rule pass/fail.",
        input_model=CheckEligibilityInput,
        output_model=CheckEligibilityOutput,
        handler=check_eligibility_handler,
        never_llm=True,
    )
    gateway.register(
        name="calculate_financials",
        description="DETERMINISTIC. Compute EMI/totals for a scheme with the pure financial engine.",
        input_model=CalculateFinancialsInput,
        output_model=CalculateFinancialsOutput,
        handler=calculate_financials_handler,
        never_llm=True,
    )
    gateway.register(
        name="match_partners",
        description="DETERMINISTIC. Match demo partner records by scheme_id + state.",
        input_model=MatchPartnersInput,
        output_model=MatchPartnersOutput,
        handler=match_partners_handler,
        never_llm=True,
    )


def register_rag_tools(gateway: ToolGateway, settings: Settings) -> None:
    async def rag_search_handler(payload: RagSearchInput) -> RagSearchOutput:
        service = get_rag_service()
        chunks = await service.search(payload.query, scheme_id=payload.scheme_id, top_k=payload.top_k)
        return RagSearchOutput(chunks=chunks, provider=service.provider, mock_mode=service.mock_mode)

    async def verify_claims_handler(payload: VerifyClaimsInput) -> VerifyClaimsOutput:
        verdicts = await verify_claims(payload.claims, payload.chunks)
        return VerifyClaimsOutput(verdicts=verdicts)

    async def explanation_handler(payload: GenerateExplanationInput) -> GenerateExplanationOutput:
        explanation = await generate_explanation(payload.payload)
        return GenerateExplanationOutput(explanation=explanation)

    gateway.register(
        name="rag_search",
        description="Retrieve scheme document chunks (with provenance + freshness) from the vector store.",
        input_model=RagSearchInput,
        output_model=RagSearchOutput,
        handler=rag_search_handler,
        mock_mode=not settings.pinecone_enabled,
    )
    gateway.register(
        name="verify_claims",
        description="Classify each claim as SUPPORTED / PARTIALLY_SUPPORTED / INSUFFICIENT_EVIDENCE / CONFLICTING_EVIDENCE.",
        input_model=VerifyClaimsInput,
        output_model=VerifyClaimsOutput,
        handler=verify_claims_handler,
        mock_mode=not settings.groq_enabled,
    )
    gateway.register(
        name="generate_explanation",
        description="Explain evaluated results in plain language with a numeric guard (LLM or template fallback).",
        input_model=GenerateExplanationInput,
        output_model=GenerateExplanationOutput,
        handler=explanation_handler,
        mock_mode=not settings.groq_enabled,
    )


def register_research_tools(gateway: ToolGateway, settings: Settings) -> None:
    """External research (Tavily) - registered by the research module in Phase 5."""
    from app.research.tavily_client import tavily_search

    async def external_research_handler(payload: ExternalResearchInput) -> ExternalResearchOutput:
        bundle = await tavily_search(payload.query, max_results=payload.max_results)
        return ExternalResearchOutput(items=bundle.items, provider=bundle.provider, mock_mode=bundle.mock_mode)

    gateway.register(
        name="external_research",
        description="Query Tavily for external web research. Results are ALWAYS labelled external and never merged with internal evidence.",
        input_model=ExternalResearchInput,
        output_model=ExternalResearchOutput,
        handler=external_research_handler,
        mock_mode=not settings.tavily_enabled,
    )


def build_gateway(settings: Settings | None = None) -> ToolGateway:
    settings = settings or get_settings()
    gateway = ToolGateway(default_timeout_seconds=settings.tool_timeout_seconds)
    register_core_tools(gateway, settings)
    register_rag_tools(gateway, settings)
    try:
        register_research_tools(gateway, settings)
    except ModuleNotFoundError:
        logger.info("external_research tool not registered (research module absent)")
    return gateway


_gateway: ToolGateway | None = None


def get_gateway() -> ToolGateway:
    global _gateway
    if _gateway is None:
        _gateway = build_gateway()
    return _gateway


def set_gateway(gateway: ToolGateway | None) -> None:
    global _gateway
    _gateway = gateway
