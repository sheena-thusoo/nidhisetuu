"""Phase 6 tests: JWT auth + the five-state application lifecycle over the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def client(mongo_db, fresh_rag_service):
    return TestClient(create_app())


def _register(client, email="applicant@example.com", password="s3cretpass!", admin=False, name=None):
    payload = {"email": email, "password": password, "name": name}
    if admin:
        payload["admin_invite_code"] = ADMIN_KEY
    return client.post("/auth/register", json=payload)


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------- auth


def test_register_login_me_roundtrip(client):
    response = _register(client, name="Applicant One")
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "applicant@example.com"
    assert body["user"]["is_admin"] is False
    assert "password_hash" not in body["user"]

    login = client.post("/auth/login", json={"email": "applicant@example.com", "password": "s3cretpass!"})
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/auth/me", headers=_auth_header(token))
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "applicant@example.com"


def test_register_rejects_duplicate_email(client):
    assert _register(client).status_code == 201
    assert _register(client).status_code == 409


def test_register_rejects_short_password(client):
    response = _register(client, password="short")
    assert response.status_code == 422


def test_register_with_bad_invite_code_is_403(client):
    response = _register(client, admin=False)
    payload = {"email": "x@example.com", "password": "s3cretpass!", "admin_invite_code": "nope"}
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 403


def test_admin_invite_code_creates_admin(client):
    response = _register(client, email="officer@example.com", admin=True)
    assert response.status_code == 201
    assert response.json()["user"]["is_admin"] is True


def test_login_wrong_password_401(client):
    _register(client)
    response = client.post("/auth/login", json={"email": "applicant@example.com", "password": "wrong-password"})
    assert response.status_code == 401


def test_tampered_and_missing_tokens_rejected(client):
    assert client.get("/auth/me").status_code == 401
    tampered = "eyJhbGciOiJIUzI1NiJ9.tampered.sig"
    assert client.get("/auth/me", headers=_auth_header(tampered)).status_code == 401


# ------------------------------------------------------------- applications


def test_full_application_lifecycle_with_admin_decisions(client):
    applicant_token = _register(client, email="applicant2@example.com").json()["access_token"]
    officer_token = _register(client, email="officer2@example.com", admin=True).json()["access_token"]

    created = client.post(
        "/applications",
        json={"scheme_id": "demo_nfsdc_education_loan", "facts": {"annual_income": 80000, "category": "SC", "location_type": "rural", "age": 22, "course_level": "undergraduate"}},
        headers=_auth_header(applicant_token),
    )
    assert created.status_code == 201
    application = created.json()
    assert application["status"] == "DRAFT"
    assert application["eligibility_snapshot"]["status"] == "potentially_eligible"
    assert application["eligibility_snapshot"]["computed_by"] == "app/services/eligibility.py"
    app_id = application["application_id"]

    # owner transitions
    assert client.post(f"/applications/{app_id}/submit", headers=_auth_header(applicant_token)).json()["status"] == "SUBMITTED"
    # officer moves it forward
    reviewed = client.post(f"/applications/{app_id}/review", json={"note": "documents verified"}, headers=_auth_header(officer_token))
    assert reviewed.json()["status"] == "UNDER_REVIEW"
    approved = client.post(f"/applications/{app_id}/decision", json={"decision": "approve"}, headers=_auth_header(officer_token))
    assert approved.json()["status"] == "APPROVED"

    # the full audit trail is queryable by the owner
    audit = client.get(f"/applications/{app_id}/audit", headers=_auth_header(applicant_token))
    events = audit.json()["events"]
    assert [e["to_status"] for e in events] == ["DRAFT", "SUBMITTED", "UNDER_REVIEW", "APPROVED"]
    assert all(e["actor_id"] for e in events)


def test_invalid_transitions_are_409(client):
    token = _register(client, email="applicant3@example.com").json()["access_token"]
    app_id = client.post(
        "/applications",
        json={"scheme_id": "demo_term_loan", "facts": {}},
        headers=_auth_header(token),
    ).json()["application_id"]
    # cannot approve a DRAFT
    response = client.post(
        "/applications",  # placeholder to keep flake quiet
    )
    officer_token = _register(client, email="officer3@example.com", admin=True).json()["access_token"]
    response = client.post(f"/applications/{app_id}/decision", json={"decision": "approve"}, headers=_auth_header(officer_token))
    assert response.status_code == 409
    # cannot submit twice
    assert client.post(f"/applications/{app_id}/submit", headers=_auth_header(token)).status_code == 200
    assert client.post(f"/applications/{app_id}/submit", headers=_auth_header(token)).status_code == 409


def test_ownership_is_enforced(client):
    owner_token = _register(client, email="owner@example.com").json()["access_token"]
    stranger_token = _register(client, email="stranger@example.com").json()["access_token"]
    officer_token = _register(client, email="officer4@example.com", admin=True).json()["access_token"]

    app_id = client.post(
        "/applications",
        json={"scheme_id": "demo_micro_finance", "facts": {"years_in_operation": 3, "gender": "female"}},
        headers=_auth_header(owner_token),
    ).json()["application_id"]

    # a stranger cannot see it (404, existence not revealed)
    assert client.get(f"/applications/{app_id}", headers=_auth_header(stranger_token)).status_code == 404
    # a non-admin cannot review it
    assert client.post(f"/applications/{app_id}/review", json={}, headers=_auth_header(stranger_token)).status_code == 403
    # the owner cannot review their own application (admin action)
    assert client.post(f"/applications/{app_id}/review", json={}, headers=_auth_header(owner_token)).status_code == 403
    # an admin CAN see it
    assert client.get(f"/applications/{app_id}", headers=_auth_header(officer_token)).status_code == 200


def test_unknown_scheme_rejected_with_422(client):
    token = _register(client, email="applicant4@example.com").json()["access_token"]
    response = client.post(
        "/applications",
        json={"scheme_id": "not_a_scheme", "facts": {}},
        headers=_auth_header(token),
    )
    assert response.status_code == 422


def test_applications_require_auth(client):
    assert client.post("/applications", json={"scheme_id": "demo_term_loan", "facts": {}}).status_code == 401
    assert client.get("/applications").status_code == 401
