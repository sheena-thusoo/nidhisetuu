"""Partner matching is a deterministic filter over clearly-labelled demo data."""

from __future__ import annotations

from app.services.partners import get_partner_service


def test_dataset_is_labelled_demo():
    service = get_partner_service()
    assert service.dataset.demo_data is True
    assert "DEMO" in service.dataset.disclaimer.upper()
    assert all(p.is_demo for p in service.dataset.partners)
    assert len(service.dataset.partners) == 8


def test_match_by_scheme_and_state():
    service = get_partner_service()
    matches = service.match("demo_nfsdc_education_loan", "Tamil Nadu")
    ids = [m.partner.partner_id for m in matches]
    assert ids == ["P004", "P008"]  # Tamil Nadu partner + all-India wildcard desk


def test_state_matching_is_case_insensitive():
    service = get_partner_service()
    assert [m.partner.partner_id for m in service.match("demo_micro_finance", "bihar")] == ["P001", "P008"]


def test_unknown_state_matches_only_wildcard():
    service = get_partner_service()
    assert [m.partner.partner_id for m in service.match("demo_term_loan", "Goa")] == ["P008"]


def test_no_state_matches_only_wildcard():
    service = get_partner_service()
    assert [m.partner.partner_id for m in service.match("demo_micro_finance", None)] == ["P008"]


def test_scheme_without_partner_and_no_wildcard_fallback():
    service = get_partner_service()
    assert service.match("demo_scheme_that_does_not_exist", "Bihar") == []


def test_match_many_groups_by_scheme():
    service = get_partner_service()
    grouped = service.match_many(["demo_micro_finance", "demo_term_loan"], "Bihar")
    assert set(grouped) == {"demo_micro_finance", "demo_term_loan"}
    assert [m.partner.partner_id for m in grouped["demo_micro_finance"]] == ["P001", "P008"]
    assert [m.partner.partner_id for m in grouped["demo_term_loan"]] == ["P001", "P008"]


def test_matched_on_records_basis():
    service = get_partner_service()
    match = service.match("demo_term_loan", "Bihar")[0]
    assert match.matched_on == {"scheme_id": "demo_term_loan", "state": "Bihar"}
