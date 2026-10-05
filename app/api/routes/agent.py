"""Agentic endpoints: POST /schemes/match (match flow) and POST /research (research flow).

NOTE: no `from __future__ import annotations` here - slowapi's limit decorator wraps
the endpoint and FastAPI would then resolve the string annotations against slowapi's
module namespace, silently turning typed body params into query params (422s).
Python 3.11 handles the union syntax natively at runtime.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.ai.graph import initial_state, run_flow
from app.api.deps import SettingsDep
from app.api.ratelimit import limiter
from app.config import get_settings
from app.schemas.api import MatchRequest, MatchResponse, ResearchRequest, ResearchResponse
from app.tools.registry import get_gateway
from app.utils.ids import new_request_id
from app.utils.logging import get_request_id, log_event

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


def _mock_flags() -> dict[str, bool]:
    return get_settings().mock_mode_flags()


@router.post("/schemes/match", response_model=MatchResponse)
@limiter.limit("30/minute")
async def match_schemes(request: Request, payload: MatchRequest):
    """Match applicant facts against every demo scheme, with evidence verification.

    Deterministic eligibility/financials run first; RAG evidence is retrieved and
    verified per candidate scheme; external web research runs ONLY as a labelled
    fallback when internal evidence is insufficient (or never, if disabled).
    """
    if not (payload.query and payload.query.strip()) and not payload.facts:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="provide `query` (free text), `facts`, or both",
        )
    request_id = get_request_id() or new_request_id()  # reuse the middleware-stamped id
    gateway = get_gateway()
    state = initial_state(
        mode="match",
        request_id=request_id,
        query=payload.query,
        facts=payload.facts or None,
        scheme_ids=payload.scheme_ids,
        top_k=payload.top_k,
        allow_external_fallback=payload.allow_external_fallback,
    )
    final = await run_flow(gateway, state)

    extraction = final.get("extraction")
    explanation = final.get("explanation")
    log_event(
        logger, logging.INFO, "match response assembled",
        request_id=request_id, candidates=len(final.get("candidate_ids", []) or []),
    )
    return MatchResponse(
        request_id=request_id,
        facts=final.get("facts") or {},
        unresolved_questions=(extraction.unresolved_questions if extraction else []),
        evaluations=[ev.model_dump(mode="json") for ev in final.get("evaluations") or []],
        candidate_ids=final.get("candidate_ids") or [],
        financials={k: v.model_dump(mode="json") for k, v in (final.get("financials") or {}).items()},
        partners={k: [p.model_dump(mode="json") for p in v] for k, v in (final.get("partners") or {}).items()},
        verdicts_by_scheme={
            k: [v.model_dump(mode="json") for v in vs]
            for k, vs in (final.get("verdicts_by_scheme") or {}).items()
        },
        evidence_sufficient=bool(final.get("evidence_sufficient")),
        external_research=[item.model_dump(mode="json") for item in final.get("external_research") or []],
        external_provider=final.get("external_provider") or "unavailable",
        external_mock_mode=bool(final.get("external_mock_mode", True)),
        fallback_used=bool(final.get("fallback_used")),
        explanation=explanation.model_dump(mode="json") if explanation else {},
        warnings=final.get("warnings") or [],
        errors=final.get("errors") or [],
        tool_trace=final.get("tool_trace") or [],
        mock_mode=_mock_flags(),
    )


@router.post("/research", response_model=ResearchResponse)
@limiter.limit("15/minute")
async def research(request: Request, payload: ResearchRequest):
    """Evidence-backed research: internal RAG chunks + external web results, verified.

    Internal evidence and external web results are reported SEPARATELY and never
    merged; every external item is labelled and caveated.
    """
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="`query` is required")
    request_id = get_request_id() or new_request_id()  # reuse the middleware-stamped id
    gateway = get_gateway()
    state = initial_state(
        mode="research",
        request_id=request_id,
        query=query,
        top_k=payload.top_k,
    )
    final = await run_flow(gateway, state)
    explanation = final.get("explanation")
    return ResearchResponse(
        request_id=request_id,
        results=final.get("research_verdicts") or [],
        external_research=[item.model_dump(mode="json") for item in final.get("external_research") or []],
        external_provider=final.get("external_provider") or "unavailable",
        external_mock_mode=bool(final.get("external_mock_mode", True)),
        explanation=explanation.model_dump(mode="json") if explanation else {},
        warnings=final.get("warnings") or [],
        errors=final.get("errors") or [],
        tool_trace=final.get("tool_trace") or [],
        mock_mode=_mock_flags(),
    )
