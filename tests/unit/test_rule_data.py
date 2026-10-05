"""Scheme rule data must load, validate, and stay internally consistent."""

from __future__ import annotations

import pytest

from app.rules.loader import RuleDataError, RuleRepository


def test_repository_loads_all_demo_schemes(rule_repo):
    ids = sorted(rule_repo.ids())
    assert ids == [
        "demo_micro_finance",
        "demo_nbcfdc_education_loan",
        "demo_nfsdc_education_loan",
        "demo_term_loan",
    ]


def test_every_scheme_is_labelled_demo(rule_repo):
    for scheme in rule_repo.all():
        assert scheme.demo_data is True
        assert "demo" in (scheme.source.type or "")
        assert scheme.description


def test_nfsdc_income_thresholds_are_the_expected_split(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    income_rule = [r for r in scheme.eligibility_rules if r.rule_id == "income_limit"][0]
    assert income_rule.operator == "threshold_map"
    assert income_rule.thresholds == {"rural": 81000, "urban": 103000, "default": 103000}
    assert income_rule.location_field == "location_type"


def test_default_tenure_is_always_an_option(rule_repo):
    for scheme in rule_repo.all():
        assert scheme.financials.default_tenure_months in scheme.financials.tenure_months_options


def test_claims_are_documented_for_every_scheme(rule_repo):
    for scheme in rule_repo.all():
        assert scheme.claims, f"{scheme.scheme_id} has no claims to verify against retrieved evidence"


def test_invalid_rule_file_raises_rule_data_error(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text('{"scheme_id": "x", "name": "x", "eligibility_rules": [], "financials": {}}', encoding="utf-8")
    with pytest.raises(RuleDataError):
        RuleRepository(tmp_path)


def test_unknown_operator_in_rule_file_is_rejected(tmp_path):
    bad = tmp_path / "bad_operator.json"
    bad.write_text(
        """
        {
          "scheme_id": "x", "name": "x",
          "eligibility_rules": [{"rule_id": "r1", "field": "age", "operator": "magic"}],
          "financials": {
            "max_principal": 1000, "interest_rate_annual_percent": 10,
            "tenure_months_options": [12], "default_tenure_months": 12
          }
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(RuleDataError):
        RuleRepository(tmp_path)
