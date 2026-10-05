"""Graph flow tests using injected in-memory RAG (mock mode everywhere).

The singleton RAG service is replaced with a fresh RagService over an InMemoryVectorStore
so tests do not depend on module import order or earlier ingestion state.
"""

from __future__ import annotations

import pytest

from app.ai.graph import (
    MIN_EVIDENCE_STATUSES,
    build_graph,
    initial_state,
    run_flow,
)
from app.config import get_settings
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.retrieval import RagService, set_rag_service
from app.rag.vector_store import InMemoryVectorStore
from app.tools.registry import build_gateway


@pytest.fixture
async def rag_with_demo_docs():
    """Fresh in-memory RagService pre-indexed with one NFSDC document."""
    from app.schemas.rag import RawChunk
    from app.utils.timeutil import utcnow

    service = RagService(store=InMemoryVectorStore(), embedder=HashEmbeddingProvider(dim=256), top_k=4)
    chunks = [
        RawChunk(chunk_index=0, text=(
            "Eligibility. The applicant must belong to the Scheduled Caste (SC) category. "
            "The annual family income limit is INR 81,000 for rural applicants and INR 1,03,000 "
            "for urban applicants. Age must be between 18 and 35 years."
        )),
        RawChunk(chunk_index=1, text=(
            "Financials. The education loan interest rate is 10% per annum. The maximum loan "
            "amount is INR 15,00,000. Tenure options are 12, 24, 36, 48, 60 and 72 months."
        )),
    ]
    await service.index_chunks(
        scheme_id="demo_nfsdc_education_loan",
        scheme_version=1,
        content_hash="ab" * 32,
        source_url="https://www.nsfdc.nic.in/",
        source_type="file",
        document_file="demo_nfsdc_education_loan.txt",
        retrieved_at=utcnow().isoformat(),
        raw_chunks=chunks,
        demo_data=True,
    )
    set_rag_service(service)
    yield service
    set_rag_service(None)


FACTS = {
    "annual_income": 81000,
    "category": "SC",
    "location_type": "rural",
    "age": 19,
    "course_level": "undergraduate",
    "loan_amount_requested": 80000,
}


async def test_gateway_registers_expected_tools_and_mock_flags(settings):
    gateway = build_gateway()
    names = gateway.tool_names()
    assert {
        "check_eligibility",
        "calculate_financials",
        "match_partners",
        "extract_intake",
        "rag_search",
        "verify_claims",
        "generate_explanation",
    } <= set(names)
    descriptors = {d["name"]: d for d in gateway.descriptors()}
    assert descriptors["check_eligibility"]["deterministic"] is True
    assert descriptors["calculate_financials"]["deterministic"] is True
    assert descriptors["rag_search"]["mock_mode"] is True  # no PINECONE key in tests


async def test_match_flow_potentially_eligible_with_financials_and_partners(rag_with_demo_docs):
    gateway = build_gateway()
    final = await run_flow(
        gateway,
        initial_state(
            mode="match",
            request_id="graph-test-1",
            query="rural SC student needs 80000",
            facts=dict(FACTS),
            top_k=3,
        ),
    )

    # deterministic eligibility across all demo schemes, sorted by status priority
    statuses = {e.scheme_id: e.eligibility.status for e in final["evaluations"]}
    assert statuses["demo_nfsdc_education_loan"] == "potentially_eligible"
    assert statuses["demo_nbcfdc_education_loan"] == "not_eligible"  # SC is not its category
    assert final["candidate_ids"][0] == "demo_nfsdc_education_loan"

    # financials only for candidates, always from the deterministic engine
    fin = final["financials"]["demo_nfsdc_education_loan"]
    assert fin.deterministic is True
    assert fin.engine == "app/services/financials.py"
    assert fin.monthly_emi.as_tuple().exponent == -2  # money is 2dp
    assert fin.loan_amount == 80000

    # deterministic partner match attached
    assert final["partners"]["demo_nfsdc_education_loan"], "expected at least one demo partner"

    # RAG was queried per candidate and evidence was verified lexically
    assert final["verdicts_by_scheme"]["demo_nfsdc_education_loan"]
    trace_tools = [entry["tool"] for entry in final["tool_trace"]]
    for expected in ("extract_intake", "check_eligibility", "calculate_financials",
                     "match_partners", "rag_search", "verify_claims", "generate_explanation"):
        assert expected in trace_tools

    # explanation produced from the deterministic payload (template in mock mode)
    assert final["explanation"].provider == "template"
    assert "NSFDC Education Loan" in final["explanation"].text


async def test_income_boundary_81000_rural_is_potentially_eligible():
    from app.rules.loader import get_rule_repository
    from app.services.eligibility import match_schemes

    evaluations = match_schemes(get_rule_repository(), dict(FACTS))
    nfsrc = [e for e in evaluations if e.scheme_id == "demo_nfsdc_education_loan"][0]
    assert nfsrc.eligibility.status == "potentially_eligible"
    over = dict(FACTS, annual_income=81001)
    over_eval = match_schemes(get_rule_repository(), over, scheme_ids=["demo_nfsdc_education_loan"])[0]
    assert over_eval.eligibility.status == "not_eligible"


async def test_no_candidates_skips_retrieval_and_explains():
    gateway = build_gateway()
    final = await run_flow(
        gateway,
        initial_state(
            mode="match",
            request_id="graph-test-2",
            query="urban OBC identity document help",
            facts={"category": "OBC", "location_type": "urban", "age": 17},
            top_k=3,
        ),
    )
    # age 17 fails both education schemes' age rules and micro-finance min age;
    # term loan requires business facts -> insufficient_data or not_eligible mix
    eligible = [e for e in final["evaluations"] if e.eligibility.status == "potentially_eligible"]
    assert eligible == []
    assert final["candidate_ids"] == []
    assert "rag_search" not in [t["tool"] for t in final["tool_trace"]]
    assert final["explanation"].text


async def test_insufficient_data_when_critical_field_is_missing():
    gateway = build_gateway()
    final = await run_flow(
        gateway,
        initial_state(
            mode="match",
            request_id="graph-test-3",
            query="",
            facts={"category": "SC", "location_type": "rural", "age": 20, "course_level": "undergraduate"},
            # annual_income missing -> NFSDC must be insufficient_data, not guessed
            top_k=2,
        ),
    )
    nfsrc = [e for e in final["evaluations"] if e.scheme_id == "demo_nfsdc_education_loan"][0]
    assert nfsrc.eligibility.status == "insufficient_data"
    assert "annual_income" in nfsrc.eligibility.missing_fields


async def test_rag_search_failure_degrades_with_warning():
    from app.rag.retrieval import RagService

    class _BrokenService:
        provider = "broken"
        mock_mode = True

        async def search(self, *args, **kwargs):
            raise RuntimeError("store down")

        async def index_chunks(self, *args, **kwargs):  # pragma: no cover
            return []

    set_rag_service(_BrokenService())  # type: ignore[arg-type]
    try:
        gateway = build_gateway()
        final = await run_flow(
            gateway,
            initial_state(
                mode="match",
                request_id="graph-test-4",
                query="rural SC student",
                facts=dict(FACTS),
                allow_external_fallback=True,
                top_k=2,
            ),
        )
        assert any("rag_search failed" in w or "rag_search" in w for w in final["warnings"])
        assert final["external_mock_mode"] is True  # external fallback attempted (mock/unregistered)
    finally:
        set_rag_service(None)


async def test_research_mode_always_queries_rag_and_external(rag_with_demo_docs):
    gateway = build_gateway()
    final = await run_flow(
        gateway,
        initial_state(
            mode="research",
            request_id="graph-test-5",
            query="NSFDC education loan interest rate and rural income limit",
            top_k=3,
        ),
    )
    assert len(final["chunks_global"]) > 0
    verdict_rows = final["research_verdicts"]
    assert verdict_rows, "research mode must verify each retrieved chunk"
    row = verdict_rows[0]
    assert {"chunk_id", "scheme_id", "verification_status", "freshness", "snippet", "claims"} <= set(row)
    assert row["verification_status"] in {"SUPPORTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"}
    # external tool is always attempted in research mode even without Tavily
    assert "external_research" in [t["tool"] for t in final["tool_trace"]]
    assert final["external_mock_mode"] is True
    assert final["fallback_used"] is False  # research mode is not a "fallback"


async def test_graph_routes_present_in_compiled_graph(rag_with_demo_docs):
    graph = build_graph(build_gateway())
    node_names = set(graph.get_graph().nodes)
    assert {"intake", "eligibility", "retrieval", "evidence", "external_research_node", "explain"} <= node_names


async def test_min_evidence_statuses_are_the_two_positive_statuses():
    assert "SUPPORTED" in MIN_EVIDENCE_STATUSES
    assert "PARTIALLY_SUPPORTED" in MIN_EVIDENCE_STATUSES
    assert "CONFLICTING_EVIDENCE" not in MIN_EVIDENCE_STATUSES
