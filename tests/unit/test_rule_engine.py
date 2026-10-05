"""Rule engine boundary tests - the deterministic core must be exact.

Covers: exact threshold, one unit over/under, missing fields, the rural/urban
threshold split, advisory (non-critical) rules, and malformed operator data.
"""

from __future__ import annotations

from app.rules.engine import evaluate, evaluate_rules, run_rule, summarize
from app.rules.operators import coerce_decimal
from app.schemas.rules import RuleSpec


def nfsdc_rule(scheme, rule_id: str) -> RuleSpec:
    for rule in scheme.eligibility_rules:
        if rule.rule_id == rule_id:
            return rule
    raise AssertionError(f"rule {rule_id} not found")


# --------------------------------------------------------------- numeric coercion


def test_coerce_decimal_variants():
    assert coerce_decimal("81,000") == 81000
    assert coerce_decimal("\u20b91,03,000") == 103000
    assert coerce_decimal("2.5 lakh") == 250000
    assert coerce_decimal("1 crore") == 10000000
    assert coerce_decimal(81000) == 81000
    assert coerce_decimal("81000+") == 81000
    assert coerce_decimal("not-a-number") is None
    assert coerce_decimal(True) is None  # booleans are not numbers here


# ----------------------------------------------------- rural/urban income split


def test_income_exact_threshold_rural_passes(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    result = run_rule(rule, {"annual_income": 81000, "location_type": "rural"})
    assert result.status == "pass"
    assert result.actual == 81000
    assert result.threshold == 81000
    assert result.threshold_variant == "rural"
    assert "81000 <= 81000" in result.comparison


def test_income_one_unit_over_rural_fails(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    result = run_rule(rule, {"annual_income": 81001, "location_type": "rural"})
    assert result.status == "fail"
    assert result.passed is False
    assert result.threshold == 81000
    assert result.threshold_variant == "rural"


def test_income_one_unit_under_rural_passes(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    assert run_rule(rule, {"annual_income": 80999, "location_type": "rural"}).status == "pass"


def test_income_boundary_urban(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    assert run_rule(rule, {"annual_income": 103000, "location_type": "urban"}).status == "pass"
    assert run_rule(rule, {"annual_income": 103001, "location_type": "urban"}).status == "fail"


def test_same_income_different_location_flips_result(rule_repo):
    """Explicit rural/urban split boundary: INR 90,000 is over the rural limit, under the urban one."""
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    rural = run_rule(rule, {"annual_income": 90000, "location_type": "rural"})
    urban = run_rule(rule, {"annual_income": 90000, "location_type": "urban"})
    assert rural.status == "fail" and rural.threshold == 81000
    assert urban.status == "pass" and urban.threshold == 103000


def test_missing_location_falls_back_to_default_threshold(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "income_limit")
    ok = run_rule(rule, {"annual_income": 103000})
    over = run_rule(rule, {"annual_income": 103001})
    assert ok.status == "pass" and ok.threshold_variant == "default"
    assert over.status == "fail" and over.threshold == 103000


def test_threshold_map_single_variant_without_default(rule_repo):
    spec = RuleSpec(
        rule_id="single_variant",
        field="annual_income",
        operator="threshold_map",
        direction="lte",
        thresholds={"rural": 50000},
        location_field="location_type",
    )
    result = run_rule(spec, {"annual_income": 40000, "location_type": "urban"})
    assert result.status == "pass"
    assert result.threshold_variant == "rural"


def test_threshold_map_without_usable_variant_is_missing_data(rule_repo):
    spec = RuleSpec(
        rule_id="no_variant",
        field="annual_income",
        operator="threshold_map",
        thresholds={"rural": 50000, "urban": 70000},
        location_field="location_type",
    )
    result = run_rule(spec, {"annual_income": 40000, "location_type": "semi-urban"})
    assert result.status == "missing_data"
    assert "location_type" in result.missing_fields


# ------------------------------------------------------------------- missing data


def test_missing_income_reports_missing_data(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    result = run_rule(nfsdc_rule(scheme, "income_limit"), {"location_type": "rural"})
    assert result.status == "missing_data"
    assert result.passed is None
    assert result.missing_fields == ["annual_income"]


def test_unparseable_income_reports_missing_data(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    result = run_rule(nfsdc_rule(scheme, "income_limit"), {"annual_income": "lots", "location_type": "rural"})
    assert result.status == "missing_data"
    assert result.note == "unparseable numeric value"


def test_overall_status_insufficient_data_when_only_data_missing(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    facts = {"category": "SC", "age": 22, "course_level": "diploma"}  # income + location missing
    eligibility = evaluate(scheme.scheme_id, scheme.version, scheme.eligibility_rules, facts)
    assert eligibility.status == "insufficient_data"
    assert "annual_income" in eligibility.missing_fields
    assert eligibility.failed_count == 0


# ----------------------------------------------------------- category / age edges


def test_category_case_insensitive_pass(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    result = run_rule(nfsdc_rule(scheme, "category"), {"category": "sc"})
    assert result.status == "pass"


def test_category_wrong_value_fails_whole_scheme(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    facts = {
        "category": "General",
        "annual_income": 50000,
        "location_type": "rural",
        "age": 22,
        "course_level": "undergraduate",
    }
    eligibility = evaluate(scheme.scheme_id, scheme.version, scheme.eligibility_rules, facts)
    assert eligibility.status == "not_eligible"
    assert eligibility.failed_count == 1
    failing = [r for r in eligibility.reasons if r.status == "fail"][0]
    assert failing.rule_id == "category"
    assert failing.actual == "General"


def test_age_boundaries(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    rule = nfsdc_rule(scheme, "age_limit")
    assert run_rule(rule, {"age": 18}).status == "pass"
    assert run_rule(rule, {"age": 35}).status == "pass"
    assert run_rule(rule, {"age": 36}).status == "fail"
    assert run_rule(rule, {"age": 17}).status == "fail"


def test_bool_equality_rule(rule_repo):
    scheme = rule_repo.get("demo_term_loan")
    rule = nfsdc_rule(scheme, "no_existing_default")
    assert run_rule(rule, {"existing_loan": False}).status == "pass"
    assert run_rule(rule, {"existing_loan": True}).status == "fail"
    assert run_rule(rule, {}).status == "missing_data"


# --------------------------------------------------------------- advisory rules


def test_non_critical_rule_failure_does_not_flip_verdict(rule_repo):
    scheme = rule_repo.get("demo_micro_finance")
    facts = {
        "age": 30,
        "annual_income": 100000,
        "location_type": "rural",
        "years_in_operation": 3,
        "gender": "male",
    }
    eligibility = evaluate(scheme.scheme_id, scheme.version, scheme.eligibility_rules, facts)
    priority = [r for r in eligibility.reasons if r.rule_id == "priority_group"][0]
    assert priority.status == "fail"
    assert priority.critical is False
    assert eligibility.status == "potentially_eligible"  # advisory rule does not disqualify


def test_unknown_operator_is_skipped_not_crash():
    spec = RuleSpec(rule_id="weird", field="age", operator="definitely_not_real")
    result = run_rule(spec, {"age": 30})
    assert result.status == "skipped"
    assert "unknown operator" in (result.note or "")


# ------------------------------------------------------------------ summary math


def test_summary_counts_and_potentially_eligible(rule_repo):
    scheme = rule_repo.get("demo_nfsdc_education_loan")
    facts = {
        "category": "SC",
        "annual_income": 81000,
        "location_type": "rural",
        "age": 22,
        "course_level": "undergraduate",
    }
    results = evaluate_rules(scheme.eligibility_rules, facts)
    summary = summarize(scheme.scheme_id, scheme.version, results)
    assert summary.status == "potentially_eligible"
    assert summary.passed_count == 4
    assert summary.failed_count == 0
    assert summary.missing_count == 0
    assert summary.disclaimer  # fixed disclaimer always attached


def test_empty_rule_set_is_potentially_eligible():
    summary = evaluate("empty_scheme", 1, [], {})
    assert summary.status == "potentially_eligible"
    assert summary.reasons == []
