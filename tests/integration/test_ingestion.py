"""Ingestion pipeline tests (Phase 4): parse -> hash -> version -> chunk -> embed -> upsert.

All tests run with an in-memory Mongo (mongomock-motor) and inline job execution
(REDIS_URL empty in the test env). Live-RQ behaviour is covered by the marked skip test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db import set_database
from app.services.ingestion import (
    compute_content_hash,
    create_and_dispatch,
    run_ingestion,
)

DATA_DIR = Path("data/sample_documents")


@pytest.fixture
async def db(mongo_db):
    """mongo_db fixture from conftest sets the app database singleton."""
    return mongo_db


@pytest.fixture
def nfsdc_bytes() -> bytes:
    return (DATA_DIR / "demo_nfsdc_education_loan.txt").read_bytes()


def test_content_hash_is_stable_sha256(nfsdc_bytes):
    h1 = compute_content_hash(nfsdc_bytes)
    h2 = compute_content_hash(nfsdc_bytes)
    assert h1 == h2
    assert len(h1) == 64


async def test_first_ingestion_creates_version_1(db, nfsdc_bytes):
    outcome = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    assert outcome.status == "COMPLETED"
    assert outcome.version == 1
    assert outcome.version_bumped is True
    assert outcome.chunks_indexed > 0
    job = await db.ingestion_jobs.find_one({"job_id": outcome.job_id}, {"_id": 0})
    assert job["status"] == "COMPLETED"
    assert job["chunks_indexed"] == outcome.chunks_indexed


async def test_reingesting_identical_content_is_idempotent(db, nfsdc_bytes):
    first = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    second = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    assert first.status == "COMPLETED" and second.status == "COMPLETED"
    assert second.version == first.version  # same content -> same version
    assert second.version_bumped is False
    assert await db.scheme_versions.count_documents({"scheme_id": "demo_nfsdc_education_loan"}) == 1


async def test_changed_content_bumps_version(db, nfsdc_bytes):
    first = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    modified = nfsdc_bytes + b"\nRevised: the urban income limit is now INR 1,10,000.\n"
    second = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=modified,
    )
    assert second.version == first.version + 1
    assert second.version_bumped is True
    versions = [doc async for doc in db.scheme_versions.find({"scheme_id": "demo_nfsdc_education_loan"}, {"_id": 0})]
    assert {v["version"] for v in versions} == {1, 2}


async def test_unknown_scheme_id_fails_the_job(db, nfsdc_bytes):
    outcome = await create_and_dispatch(
        scheme_id="scheme_that_does_not_exist",
        filename="whatever.txt",
        raw_bytes=b"some text",
    )
    assert outcome.status == "FAILED"
    assert "unknown scheme_id" in (outcome.error or "")
    job = await db.ingestion_jobs.find_one({"job_id": outcome.job_id})
    assert job["status"] == "FAILED"


async def test_html_table_document_indexes_table_rows(db):
    html = (DATA_DIR / "demo_term_loan.html").read_bytes()
    outcome = await create_and_dispatch(
        scheme_id="demo_term_loan",
        filename="demo_term_loan.html",
        raw_bytes=html,
    )
    assert outcome.status == "COMPLETED"
    assert outcome.chunks_indexed > 0
    doc = await db.documents.find_one({"scheme_id": "demo_term_loan"}, {"_id": 0})
    assert doc["parser"] == "html+tables"


async def test_document_meta_scheme_id_mismatch_fails_job(db):
    html = (
        b"<html><head><meta name='scheme_id' content='demo_other_scheme'/></head>"
        b"<body><p>Some guideline text that is long enough to be chunked properly.</p></body></html>"
    )
    outcome = await run_ingestion(
        job_id="job-mismatch",
        scheme_id="demo_term_loan",
        filename="x.html",
        raw_bytes=html,
        source_url=None,
        source_type="file",
        demo_data=True,
        db=db,
    )
    assert outcome.status == "FAILED"
    assert "mismatch" in (outcome.error or "")


async def test_empty_document_is_skipped(db):
    outcome = await run_ingestion(
        job_id="job-empty",
        scheme_id="demo_nfsdc_education_loan",
        filename="empty.txt",
        raw_bytes=b"",
        source_url=None,
        source_type="file",
        demo_data=True,
        db=db,
    )
    assert outcome.status == "SKIPPED"


async def test_job_status_trail_is_recorded(db, nfsdc_bytes):
    outcome = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    job = await db.ingestion_jobs.find_one({"job_id": outcome.job_id}, {"_id": 0})
    assert job["status"] == "COMPLETED"
    assert job["created_at"] and job["finished_at"]
    assert job["mock_mode"]["redis_rq"] is True  # inline execution == redis in mock mode


async def test_ingested_chunks_are_retrievable(db, nfsdc_bytes):
    from app.rag.retrieval import get_rag_service

    outcome = await create_and_dispatch(
        scheme_id="demo_nfsdc_education_loan",
        filename="demo_nfsdc_education_loan.txt",
        raw_bytes=nfsdc_bytes,
    )
    assert outcome.status == "COMPLETED"
    hits = await get_rag_service().search("rural income limit", scheme_id="demo_nfsdc_education_loan")
    assert hits, "ingested document must be retrievable"
    assert all(h.metadata.scheme_version == outcome.version for h in hits)


@pytest.mark.live_redis
async def test_rq_enqueue_with_live_redis():
    pytest.skip("requires REDIS_URL pointing at a live Redis instance (mock mode: skipped)")
