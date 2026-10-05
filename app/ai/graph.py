"""LangGraph wiring for the two flows.

match mode:
  intake -> deterministic eligibility (+ financials, partners) -> RAG retrieval ->
  evidence verification -> (external fallback ONLY if internal evidence is insufficient) ->
  explanation

research mode:
  intake -> RAG retrieval (top-k chunks) -> evidence verification of every result ->
  Tavily external research (ALWAYS also queried; never merged with internal evidence) ->
  explanation

Model behaviour never determines eligibility or money: those nodes call deterministic tools
through the gateway and only interpret their output.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.ai.state import GraphState, add_trace, add_warning
from app.ai.explanation import generate_explanation  # noqa: F401  (kept for type readers)
from app.rag.evidence import build_claims_for_scheme, lexical_verify, verify_claims
from app.schemas.intake import IntakeExtraction
from app.schemas.rag import EvidenceClaim, EvidenceVerdict
from app.tools.gateway import ToolGateway
from app.tools.schemas import (
    CalculateFinancialsOutput,
    CheckEligibilityOutput,
    ExternalResearchOutput,
    ExtractIntakeOutput,
    GenerateExplanationOutput,
    MatchPartnersOutput,
    RagSearchOutput,
    VerifyClaimsOutput,
)
from app.utils.logging import log_event

logger = logging.getLogger(__name__)

MIN_EVIDENCE_STATUSES = {"SUPPORTED", "PARTIALLY_SUPPORTED"}


# --------------------------------------------------------------------------- nodes


def make_intake_node(gateway: ToolGateway):
    async def intake_node(state: GraphState) -> dict:
        result = await gateway.call(
            "extract_intake",
            {"text": state.get("raw_query"), "provided": state.get("provided_facts") or {}},
            request_id=state.get("request_id"),
        )
        trace = add_trace(state, result)
        warnings = list(state.get("warnings", []))
        if result.ok and isinstance(result.output, ExtractIntakeOutput):
            extraction = result.output.extraction
        else:
            warnings.append(f"intake tool failed ({result.error.error_type if result.error else 'unknown'}); using provided facts only")
            extraction = IntakeExtraction(facts=state.get("provided_facts") or {}, provider="fallback")
            extraction.warnings = list(warnings)
        return {
            "extraction": extraction,
            "facts": extraction.facts,
            "tool_trace": trace,
            "warnings": warnings,
        }

    return intake_node


def make_eligibility_node(gateway: ToolGateway):
    async def eligibility_node(state: GraphState) -> dict:
        facts = state.get("facts") or {}
        request_id = state.get("request_id")
        trace = list(state.get("tool_trace", []))
        warnings = list(state.get("warnings", []))
        errors = list(state.get("errors", []))

        result = await gateway.call(
            "check_eligibility", {"facts": facts, "scheme_ids": state.get("scheme_ids")}, request_id=request_id
        )
        trace.append(result.trace_entry())
        if result.ok and isinstance(result.output, CheckEligibilityOutput):
            evaluations = result.output.evaluations
        else:
            errors.append(f"check_eligibility failed: {result.error.message if result.error else 'unknown'}")
            evaluations = []

        candidates = [
            ev for ev in evaluations if ev.eligibility.status in ("potentially_eligible", "insufficient_data")
        ]
        candidate_ids = [ev.scheme_id for ev in candidates]

        financials: dict = {}
        refreshed: list = []
        for evaluation in evaluations:
            if evaluation.scheme_id in candidate_ids:
                fin_result = await gateway.call(
                    "calculate_financials", {"scheme_id": evaluation.scheme_id, "facts": facts}, request_id=request_id
                )
                trace.append(fin_result.trace_entry())
                if fin_result.ok and isinstance(fin_result.output, CalculateFinancialsOutput):
                    financials[evaluation.scheme_id] = fin_result.output.financials
                    evaluation = evaluation.model_copy(update={"financials": fin_result.output.financials})
                else:
                    warnings.append(
                        f"financials unavailable for {evaluation.scheme_id}: {fin_result.error.message if fin_result.error else 'unknown'}"
                    )
            refreshed.append(evaluation)

        partners: dict = {}
        if candidate_ids:
            partner_result = await gateway.call(
                "match_partners", {"scheme_ids": candidate_ids, "state": facts.get("state")}, request_id=request_id
            )
            trace.append(partner_result.trace_entry())
            if partner_result.ok and isinstance(partner_result.output, MatchPartnersOutput):
                partners = partner_result.output.matches

        return {
            "evaluations": refreshed,
            "candidate_ids": candidate_ids,
            "financials": financials,
            "partners": partners,
            "tool_trace": trace,
            "warnings": warnings,
            "errors": errors,
        }

    return eligibility_node


def _scheme_query(evaluation, claims: list[EvidenceClaim], facts: dict) -> str:
    fact_bits = " ".join(str(facts.get(key, "")) for key in ("location_type", "category", "state") if facts.get(key))
    claim_text = " ".join(claim.claim for claim in claims)[:700]
    return f"{evaluation.name}. {claim_text} {fact_bits}".strip()


def make_retrieval_node(gateway: ToolGateway):
    async def retrieval_node(state: GraphState) -> dict:
        request_id = state.get("request_id")
        trace = list(state.get("tool_trace", []))
        warnings = list(state.get("warnings", []))
        top_k = state.get("top_k") or 4
        mode = state.get("mode", "match")

        if mode == "research":
            query = state.get("raw_query") or ""
            result = await gateway.call("rag_search", {"query": query, "top_k": top_k}, request_id=request_id)
            trace.append(result.trace_entry())
            if result.ok and isinstance(result.output, RagSearchOutput):
                chunks = result.output.chunks
                if not chunks:
                    warnings.append("internal retrieval returned no chunks (is anything ingested yet?)")
            else:
                chunks = []
                warnings.append(
                    f"rag_search failed: {result.error.message if result.error else 'unknown error'}"
                )
            return {
                "chunks_global": chunks,
                "chunks_by_scheme": {},
                "claims_by_scheme": {},
                "tool_trace": trace,
                "warnings": warnings,
            }

        chunks_by_scheme: dict[str, list] = {}
        claims_by_scheme: dict[str, list[EvidenceClaim]] = {}
        for evaluation in state.get("evaluations") or []:
            if evaluation.scheme_id not in (state.get("candidate_ids") or []):
                continue
            claims = build_claims_for_scheme(evaluation)
            claims_by_scheme[evaluation.scheme_id] = claims
            query = _scheme_query(evaluation, claims, state.get("facts") or {})
            result = await gateway.call(
                "rag_search",
                {"query": query, "scheme_id": evaluation.scheme_id, "top_k": top_k},
                request_id=request_id,
            )
            trace.append(result.trace_entry())
            if result.ok and isinstance(result.output, RagSearchOutput):
                chunks = result.output.chunks
                if not chunks:
                    warnings.append(
                        f"no indexed document chunks for {evaluation.scheme_id} (ingest a document first)"
                    )
            else:
                chunks = []
                warnings.append(
                    f"rag_search failed for {evaluation.scheme_id}: "
                    f"{result.error.message if result.error else 'unknown error'}"
                )
            chunks_by_scheme[evaluation.scheme_id] = chunks
        return {
            "chunks_by_scheme": chunks_by_scheme,
            "claims_by_scheme": claims_by_scheme,
            "tool_trace": trace,
            "warnings": warnings,
        }

    return retrieval_node


def make_evidence_node(gateway: ToolGateway):
    async def evidence_node(state: GraphState) -> dict:
        request_id = state.get("request_id")
        trace = list(state.get("tool_trace", []))
        warnings = list(state.get("warnings", []))
        mode = state.get("mode", "match")

        if mode == "research":
            chunks = state.get("chunks_global") or []
            research_verdicts: list[dict] = []
            for chunk in chunks:
                claims = _chunk_claims(chunk)
                verdicts = [lexical_verify(claim, chunks) for claim in claims]
                research_verdicts.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "scheme_id": chunk.metadata.scheme_id,
                        "scheme_version": chunk.metadata.scheme_version,
                        "section": chunk.metadata.section,
                        "source_url": chunk.metadata.source_url,
                        "retrieved_at": chunk.metadata.retrieved_at,
                        "freshness": chunk.freshness.model_dump(),
                        "snippet": chunk.text[:400],
                        "score": chunk.score,
                        "claims": [v.model_dump() for v in verdicts],
                        "verification_status": _aggregate_status(verdicts),
                    }
                )
            return {"research_verdicts": research_verdicts, "tool_trace": trace, "warnings": warnings}

        verdicts_by_scheme: dict[str, list[EvidenceVerdict]] = {}
        scheme_evidence_ok: dict[str, bool] = {}
        for scheme_id, claims in (state.get("claims_by_scheme") or {}).items():
            chunks = (state.get("chunks_by_scheme") or {}).get(scheme_id) or []
            if not claims:
                continue
            result = await gateway.call(
                "verify_claims", {"claims": claims, "chunks": chunks, "scheme_id": scheme_id}, request_id=request_id
            )
            trace.append(result.trace_entry())
            if result.ok and isinstance(result.output, VerifyClaimsOutput):
                verdicts = result.output.verdicts
            else:
                warnings.append(f"evidence verification failed for {scheme_id}; using lexical fallback")
                verdicts = [lexical_verify(claim, chunks) for claim in claims]
            verdicts_by_scheme[scheme_id] = verdicts
            scheme_evidence_ok[scheme_id] = any(
                v.verification_status in MIN_EVIDENCE_STATUSES for v in verdicts
            )
        evidence_sufficient = any(scheme_evidence_ok.values()) if scheme_evidence_ok else False
        return {
            "verdicts_by_scheme": verdicts_by_scheme,
            "scheme_evidence_ok": scheme_evidence_ok,
            "evidence_sufficient": evidence_sufficient,
            "tool_trace": trace,
            "warnings": warnings,
        }

    return evidence_node


def _chunk_claims(chunk) -> list[EvidenceClaim]:
    """Derive verifiable claims from a retrieved chunk (sentences containing numbers)."""
    from app.rag.evidence import extract_numbers

    claims: list[EvidenceClaim] = []
    sentences = [s.strip() for s in chunk.text.replace("\n", " ").split(".") if s.strip()]
    for index, sentence in enumerate(sentences):
        numbers = extract_numbers(sentence)
        if not numbers or len(sentence) < 25:
            continue
        claims.append(
            EvidenceClaim(
                claim_id=f"{chunk.chunk_id}:s{index}",
                claim=sentence[:280],
                claim_type="scheme_fact",
                scheme_id=chunk.metadata.scheme_id,
                numbers=sorted(numbers),
            )
        )
        if len(claims) >= 3:
            break
    if not claims:
        claims.append(
            EvidenceClaim(
                claim_id=f"{chunk.chunk_id}:section",
                claim=(chunk.metadata.section or "Retrieved excerpt") + ": " + chunk.text[:200],
                claim_type="scheme_fact",
                scheme_id=chunk.metadata.scheme_id,
                numbers=[],
            )
        )
    return claims


_STATUS_SEVERITY = {
    "SUPPORTED": 0,
    "PARTIALLY_SUPPORTED": 1,
    "INSUFFICIENT_EVIDENCE": 2,
    "CONFLICTING_EVIDENCE": 3,
}


def _aggregate_status(verdicts: list[EvidenceVerdict]) -> str:
    if not verdicts:
        return "INSUFFICIENT_EVIDENCE"
    worst = max(verdicts, key=lambda v: _STATUS_SEVERITY[v.verification_status])
    if worst.verification_status == "CONFLICTING_EVIDENCE":
        return "CONFLICTING_EVIDENCE"
    if all(v.verification_status == "SUPPORTED" for v in verdicts):
        return "SUPPORTED"
    if any(v.verification_status in MIN_EVIDENCE_STATUSES for v in verdicts):
        return "PARTIALLY_SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"


def make_external_node(gateway: ToolGateway):
    async def external_node(state: GraphState) -> dict:
        request_id = state.get("request_id")
        trace = list(state.get("tool_trace", []))
        warnings = list(state.get("warnings", []))
        mode = state.get("mode", "match")

        if mode == "match":
            query = state.get("raw_query") or ""
            scheme_names = [ev.name for ev in (state.get("evaluations") or [])]
            if not query:
                query = " ".join(scheme_names) or "government loan scheme eligibility"
            query = f"{query} {' '.join(scheme_names)}".strip()
        else:
            query = state.get("raw_query") or ""

        result = await gateway.call("external_research", {"query": query, "max_results": 2}, request_id=request_id)
        trace.append(result.trace_entry())
        if result.ok and isinstance(result.output, ExternalResearchOutput):
            items = result.output.items
            provider = result.output.provider
            mock_mode = result.output.mock_mode
        else:
            items = []
            provider = "unavailable"
            mock_mode = True
            warnings.append(
                "external research unavailable (no TAVILY_API_KEY and/or tool not registered): "
                f"{(result.error.message if result.error else 'unknown error')}"
            )
        return {
            "external_research": items,
            "external_provider": provider,
            "external_mock_mode": mock_mode,
            "fallback_used": mode == "match",
            "tool_trace": trace,
            "warnings": warnings,
        }

    return external_node


def make_explanation_node(gateway: ToolGateway):
    async def explanation_node(state: GraphState) -> dict:
        request_id = state.get("request_id")
        trace = list(state.get("tool_trace", []))
        payload: dict = {
            "mode": state.get("mode", "match"),
            "facts": state.get("facts") or {},
            "evaluations": [ev.model_dump(mode="json") for ev in (state.get("evaluations") or [])],
            "verdicts": [
                v.model_dump(mode="json")
                for verdicts in (state.get("verdicts_by_scheme") or {}).values()
                for v in verdicts
            ],
            "research_results": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "verification_status": item.get("verification_status"),
                    "freshness": item.get("freshness"),
                    "section": item.get("section"),
                }
                for item in (state.get("research_verdicts") or [])
            ],
            "external_count": len(state.get("external_research") or []),
        }
        result = await gateway.call("generate_explanation", {"payload": payload}, request_id=request_id)
        trace.append(result.trace_entry())
        warnings = list(state.get("warnings", []))
        if result.ok and isinstance(result.output, GenerateExplanationOutput):
            explanation = result.output.explanation
            warnings.extend(explanation.warnings)
        else:
            warnings.append("explanation tool failed; using deterministic template")
            explanation = _fallback_explanation(payload)
        return {"explanation": explanation, "tool_trace": trace, "warnings": warnings}

    return explanation_node


def _fallback_explanation(payload: dict):
    from app.ai.explanation import ExplanationResult, template_explanation

    return ExplanationResult(text=template_explanation(payload), provider="template", mock_mode=True)


# ------------------------------------------------------------------------- routing


def _route_after_intake(state: GraphState) -> str:
    return "retrieval" if state.get("mode") == "research" else "eligibility"


def _route_after_eligibility(state: GraphState) -> str:
    return "retrieval" if state.get("candidate_ids") else "explain"


def _route_after_evidence(state: GraphState) -> str:
    if state.get("mode") == "research":
        return "external_research"
    if not state.get("evidence_sufficient") and state.get("allow_external_fallback", True):
        return "external_research"
    return "explain"


# --------------------------------------------------------------------------- build


def build_graph(gateway: ToolGateway):
    # Graph node names must not collide with GraphState keys: "external_research"
    # and "explanation" are state keys, so the graph nodes use suffixed names.
    builder = StateGraph(GraphState)
    builder.add_node("intake", make_intake_node(gateway))
    builder.add_node("eligibility", make_eligibility_node(gateway))
    builder.add_node("retrieval", make_retrieval_node(gateway))
    builder.add_node("evidence", make_evidence_node(gateway))
    builder.add_node("external_research_node", make_external_node(gateway))
    builder.add_node("explain", make_explanation_node(gateway))

    builder.set_entry_point("intake")
    builder.add_conditional_edges(
        "intake", _route_after_intake, {"eligibility": "eligibility", "retrieval": "retrieval"}
    )
    builder.add_conditional_edges(
        "eligibility", _route_after_eligibility, {"retrieval": "retrieval", "explain": "explain"}
    )
    builder.add_edge("retrieval", "evidence")
    builder.add_conditional_edges(
        "evidence",
        _route_after_evidence,
        {"external_research": "external_research_node", "explain": "explain"},
    )
    builder.add_edge("external_research_node", "explain")
    builder.add_edge("explain", END)
    return builder.compile()


def initial_state(
    *,
    mode: str,
    request_id: str,
    query: str | None = None,
    facts: dict | None = None,
    scheme_ids: list[str] | None = None,
    top_k: int = 4,
    allow_external_fallback: bool = True,
) -> GraphState:
    return GraphState(
        mode=mode,  # type: ignore[typeddict-item]
        request_id=request_id,
        raw_query=query or "",
        provided_facts=facts or {},
        scheme_ids=scheme_ids,
        top_k=top_k,
        allow_external_fallback=allow_external_fallback,
        tool_trace=[],
        warnings=[],
        errors=[],
    )


async def run_flow(gateway: ToolGateway, state: GraphState) -> GraphState:
    graph = build_graph(gateway)
    final_state = await graph.ainvoke(state)
    log_event(
        logger,
        logging.INFO,
        "agent flow finished",
        mode=final_state.get("mode"),
        request_id=state.get("request_id"),
        tool_calls=len(final_state.get("tool_trace", [])),
        candidates=len(final_state.get("candidate_ids", []) or []),
        warnings=len(final_state.get("warnings", [])),
    )
    return final_state  # type: ignore[return-value]
