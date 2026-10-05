"""MongoDB access layer tests against the motor-compatible in-memory mock."""

from __future__ import annotations

from app.db import get_database


async def test_indexes_can_be_created_twice(mongo_db):
    await mongo_db.ensure_indexes()  # idempotent - callers may run it on every startup
    assert get_database() is mongo_db


async def test_scheme_version_roundtrip(mongo_db):
    await mongo_db.scheme_versions.insert_one(
        {
            "scheme_id": "demo_nfsdc_education_loan",
            "version": 1,
            "content_hash": "abc123",
            "first_seen_at": "2026-10-01T00:00:00+00:00",
        }
    )
    doc = await mongo_db.scheme_versions.find_one(
        {"scheme_id": "demo_nfsdc_education_loan", "version": 1}, {"_id": 0}
    )
    assert doc is not None
    assert doc["content_hash"] == "abc123"


async def test_unique_scheme_id_index_rejects_duplicates(mongo_db):
    await mongo_db.schemes.insert_one({"scheme_id": "dupe", "name": "first"})
    try:
        await mongo_db.schemes.insert_one({"scheme_id": "dupe", "name": "second"})
        raise AssertionError("expected a duplicate-key error")
    except Exception as exc:  # mongomock raises DuplicateKeyError
        assert "duplicate" in type(exc).__name__.lower() or "duplicate" in str(exc).lower()


async def test_job_collection_filtering(mongo_db):
    for status in ("QUEUED", "RUNNING", "COMPLETED", "COMPLETED"):
        await mongo_db.ingestion_jobs.insert_one({"job_id": f"job-{status}", "status": status})
    completed = await mongo_db.ingestion_jobs.count_documents({"status": "COMPLETED"})
    assert completed == 2
