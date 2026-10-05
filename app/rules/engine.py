"""Deterministic rule engine.

Pure functions only - no I/O, no LLM, no randomness. Same facts + same rules => same result.
Three-state outcome per rule: pass / fail / missing_data (plus skipped for defensive cases).
"""

from __future__ import annotations

from typing import Any

from app.constants import DISCLAIMER
from app.rules.operators import OPERATORS, display_value, is_blank
from app.schemas.rules import EligibilityResult, RuleResult, RuleSpec


def run_rule(spec: RuleSpec, facts: dict[str, Any]) -> RuleResult:
    actual = facts.get(spec.field)

    def result(
        status: str,
        passed: bool | None,
        *,
        comparison: str = "",
        threshold: Any | None = None,
        threshold_variant: str | None = None,
        missing_fields: list[str] | None = None,
        note: str | None = None,
    ) -> RuleResult:
        return RuleResult(
            rule_id=spec.rule_id,
            description=spec.description,
            operator=spec.operator,
            status=status,  # type: ignore[arg-type]
            passed=passed,
            actual=actual,
            actual_display=display_value(actual, spec.unit),
            threshold=threshold,
            threshold_display=display_value(threshold, spec.unit),
            threshold_variant=threshold_variant,
            comparison=comparison,
            unit=spec.unit,
            missing_fields=missing_fields or [],
            note=note,
            critical=spec.critical,
            source=spec.source,
        )

    if is_blank(actual):
        return result("missing_data", None, missing_fields=[spec.field], note="field not provided")

    operator = OPERATORS.get(spec.operator)
    if operator is None:
        return result("skipped", None, note=f"unknown operator {spec.operator!r}")

    outcome = operator(actual, spec, facts)
    if outcome.passed is None:
        missing = [spec.field] + [f for f in outcome.missing_fields if f]
        return result(
            "missing_data",
            None,
            comparison=outcome.comparison,
            threshold=outcome.threshold,
            threshold_variant=outcome.threshold_variant,
            missing_fields=sorted(set(missing)),
            note=outcome.note,
        )
    return result(
        "pass" if outcome.passed else "fail",
        outcome.passed,
        comparison=outcome.comparison,
        threshold=outcome.threshold,
        threshold_variant=outcome.threshold_variant,
        note=outcome.note,
    )


def evaluate_rules(rules: list[RuleSpec], facts: dict[str, Any]) -> list[RuleResult]:
    return [run_rule(spec, facts) for spec in rules]


def summarize(
    scheme_id: str,
    scheme_version: int,
    results: list[RuleResult],
    *,
    disclaimer: str = DISCLAIMER,
) -> EligibilityResult:
    """Aggregate rule results into a three-state eligibility verdict.

    * any CRITICAL rule fails                     -> not_eligible
    * no critical failure but critical data missing -> insufficient_data
    * otherwise                                    -> potentially_eligible
    Non-critical rules are reported but never flip the verdict.
    """
    critical = [r for r in results if r.critical]
    failures = [r for r in critical if r.status == "fail"]
    misses = [r for r in critical if r.status == "missing_data"]
    skips = [r for r in results if r.status == "skipped"]

    if failures:
        status = "not_eligible"
    elif misses:
        status = "insufficient_data"
    else:
        status = "potentially_eligible"

    missing_fields = sorted({field for r in misses for field in r.missing_fields})
    return EligibilityResult(
        scheme_id=scheme_id,
        scheme_version=scheme_version,
        status=status,
        passed_count=sum(1 for r in results if r.status == "pass"),
        failed_count=sum(1 for r in results if r.status == "fail"),
        missing_count=sum(1 for r in results if r.status == "missing_data"),
        skipped_count=len(skips),
        reasons=results,
        missing_fields=missing_fields,
        disclaimer=disclaimer,
    )


def evaluate(scheme_id: str, scheme_version: int, rules: list[RuleSpec], facts: dict[str, Any]) -> EligibilityResult:
    return summarize(scheme_id, scheme_version, evaluate_rules(rules, facts))
