"""Admin API tests (Phase 4): auth gate + ingest flow over the FastAPI app.

The app is created WITHOUT running its lifespan (no real Mongo); the mongomock
database is injected through the same set_database() singleton the app uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app

DATA_DIR = Path("data/sample_documents")
ADMIN_KEY = "test-admin-key"  # set by tests/conftest.py via ADMIN_API_KEY


@pytest.fixture
def client(mongo_db, fresh_rag_service):
    app = create_app()
    return TestClient(app)  # lifespan NOT executed -> no real Mongo connection


def test_health_reports_mock_mode(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock_mode"]["groq"] is True
    assert body["mock_mode"]["pinecone"] is True


def test_ingest_requires_admin_key(client):
    response = client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_nfsdc_education_loan"},
        files={"file": ("doc.txt", b"x", "text/plain")},
    )
    assert response.status_code == 401


def test_ingest_rejects_wrong_admin_key(client):
    response = client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_nfsdc_education_loan"},
        files={"file": ("doc.txt", b"x", "text/plain")},
        headers={"X-Admin-Key": "wrong"},
    )
    assert response.status_code == 403


def test_ingest_empty_file_is_400(client):
    response = client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_nfsdc_education_loan"},
        files={"file": ("doc.txt", b"", "text/plain")},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert response.status_code == 400


def test_ingest_unknown_scheme_fails_job_not_request(client):
    response = client.post(
        "/admin/ingest",
        data={"scheme_id": "no_such_scheme"},
        files={"file": ("doc.txt", b"some content", "text/plain")},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert response.status_code == 202  # the job was accepted
    body = response.json()
    assert body["status"] == "FAILED"
    assert "unknown scheme_id" in (body["error"] or "")


async def test_ingest_sample_document_end_to_end(client, mongo_db):
    content = (DATA_DIR / "demo_nfsdc_education_loan.txt").read_bytes()
    response = client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_nfsdc_education_loan", "source_url": "https://www.nsfdc.nic.in/"},
        files={"file": ("demo_nfsdc_education_loan.txt", content, "text/plain")},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["version"] == 1
    assert body["chunks_indexed"] > 0
    assert body["executed_inline"] is True  # no Redis in the test env

    job_id = body["job_id"]
    job = client.get(f"/admin/ingest/{job_id}", headers={"X-Admin-Key": ADMIN_KEY})
    assert job.status_code == 200
    assert job.json()["status"] == "COMPLETED"

    # idempotent re-ingestion through the API
    again = client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_nfsdc_education_loan"},
        files={"file": ("demo_nfsdc_education_loan.txt", content, "text/plain")},
        headers={"X-Admin-Key": ADMIN_KEY},
    ).json()
    assert again["version"] == 1 and again["version_bumped"] is False
    assert await mongo_db.scheme_versions.count_documents({"scheme_id": "demo_nfsdc_education_loan"}) == 1


def test_job_status_404_for_unknown_job(client):
    response = client.get("/admin/ingest/does-not-exist", headers={"X-Admin-Key": ADMIN_KEY})
    assert response.status_code == 404


def test_recent_jobs_listing(client):
    client.post(
        "/admin/ingest",
        data={"scheme_id": "demo_term_loan"},
        files={"file": ("demo_term_loan.html", (DATA_DIR / "demo_term_loan.html").read_bytes(), "text/html")},
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    listing = client.get("/admin/jobs?limit=5", headers={"X-Admin-Key": ADMIN_KEY})
    assert listing.status_code == 200
    jobs = listing.json()["jobs"]
    assert jobs and jobs[0]["scheme_id"] == "demo_term_loan"
    assert "_id" not in jobs[0]


def test_tools_catalog_lists_registered_tools(client):
    response = client.get("/api/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()["tools"]}
    assert {"check_eligibility", "calculate_financials", "rag_search", "verify_claims"} <= names


def test_request_id_middleware_echoes_and_generates(client):
    response = client.get("/health", headers={"X-Request-ID": "test-rid-123"})
    assert response.headers["X-Request-ID"] == "test-rid-123"
    generated = client.get("/health")
    assert len(generated.headers["X-Request-ID"]) >= 16
