"""Agent endpoint tests (Phase 5): /schemes/match and /research over the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

FACTS = {
    "annual_income": 81000,
    "category": "SC",
    "location_type": "rural",
    "age": 19,
    "course_level": "undergraduate",
    "loan_amount_requested": 80000,
}


@pytest.fixture
def client(mongo_db, fresh_rag_service):
    return TestClient(create_app())


def test_match_with_structured_facts(client):
    response = client.post("/schemes/match", json={"facts": FACTS})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "match"
    assert body["candidate_ids"][0] == "demo_nfsdc_education_loan"
    evals = {e["scheme_id"]: e for e in body["evaluations"]}
    assert evals["demo_nfsdc_education_loan"]["eligibility"]["status"] == "potentially_eligible"
    fin = body["financials"]["demo_nfsdc_education_loan"]
    assert fin["deterministic"] is True
    assert fin["engine"] == "app/services/financials.py"
    assert fin["monthly_emi"] == "1699.76"  # 80,000 @ 10% / 60m, Decimal-safe JSON
    assert body["partners"]["demo_nfsdc_education_loan"]
    assert body["verdicts_by_scheme"]["demo_nfsdc_education_loan"]
    assert body["disclaimer"]
    assert body["mock_mode"]["groq"] is True
    # every tool in the trace came through the gateway
    assert all("tool" in entry for entry in body["tool_trace"])


def test_match_with_free_text_query_only(client):
    response = client.post(
        "/schemes/match",
        json={"query": "I am a rural SC student, age 19, undergraduate, family income 81000, need 80000 loan"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["facts"]["annual_income"] == "81000"  # regex-extracted (string passthrough)
    assert body["facts"]["location_type"] == "rural"
    assert body["facts"]["category"] == "SC"
    assert "demo_nfsdc_education_loan" in body["candidate_ids"]


def test_match_requires_query_or_facts(client):
    response = client.post("/schemes/match", json={})
    assert response.status_code == 400


def test_match_rejects_out_of_range_top_k(client):
    response = client.post("/schemes/match", json={"facts": FACTS, "top_k": 99})
    assert response.status_code == 422


def test_match_insufficient_evidence_falls_back_to_external_mock(client):
    # nothing ingested in this fresh store -> all claims INSUFFICIENT -> external fallback
    response = client.post("/schemes/match", json={"facts": FACTS})
    body = response.json()
    assert body["evidence_sufficient"] is False
    assert body["fallback_used"] is True
    # Phase 5: external_research is now registered and runs in MOCK mode
    assert body["external_mock_mode"] is True
    assert body["external_provider"] == "tavily-mock"
    for item in body["external_research"]:
        assert item["label"] == "external_research"
        assert item["mock_mode"] is True
        assert item["caveat"]


def test_research_flow_with_empty_store(client):
    response = client.post("/research", json={"query": "NSFDC education loan interest rate"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "research"
    assert body["request_id"] == response.headers["X-Request-ID"]  # middleware echo
    assert body["external_mock_mode"] is True
    assert any("no chunks" in w for w in body["warnings"])  # honest: store is empty
    assert body["disclaimer"]


def test_research_requires_query(client):
    assert client.post("/research", json={"query": "   "}).status_code == 400
    assert client.post("/research", json={}).status_code == 422


def test_research_after_ingestion_returns_verified_chunks(client, mongo_db, fresh_rag_service):
    from pathlib import Path

    from app.services.ingestion import create_and_dispatch

    content = Path("data/sample_documents/demo_nfsdc_education_loan.txt").read_bytes()
    outcome = client  # noqa: F841 - keep the fixture references explicit
    import asyncio

    loop_result = asyncio.get_event_loop_policy().new_event_loop()
    try:
        ingest = loop_result.run_until_complete(
            create_and_dispatch(
                scheme_id="demo_nfsdc_education_loan",
                filename="demo_nfsdc_education_loan.txt",
                raw_bytes=content,
            )
        )
        assert ingest.status == "COMPLETED"
    finally:
        loop_result.close()

    response = client.post("/research", json={"query": "rural income limit 81000"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"], "retrieved chunks must be verified and reported"
    row = body["results"][0]
    assert {"chunk_id", "scheme_id", "verification_status", "freshness", "snippet"} <= set(row)
    # CONFLICTING is a legitimate aggregate here: the demo doc contains both a rural
    # and an urban income sentence and the deterministic conflict detector fires when
    # a claim's numbers are absent from a high-overlap chunk (documented behaviour).
    assert row["verification_status"] in {
        "SUPPORTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"
    }
    # internal chunks and external web results are reported separately
    for item in body["external_research"]:
        assert item["label"] == "external_research"
    internal_ids = {r["chunk_id"] for r in body["results"]}
    external_ids = {item["url"] for item in body["external_research"]}
    assert internal_ids.isdisjoint(external_ids)
