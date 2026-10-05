"""Operator registry for the data-driven rule engine.

Each operator is a pure function: (actual_value, spec, facts) -> OperatorOutcome.
Rules in JSON name an operator; nothing about a scheme is hardcoded in Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from app.schemas.rules import RuleSpec
from app.utils.format import format_inr

# --------------------------------------------------------------------------- coercion


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


_SUFFIXES: dict[str, Decimal] = {
    "lakhs": Decimal(10) ** 5,
    "lakh": Decimal(10) ** 5,
    "lacs": Decimal(10) ** 5,
    "lac": Decimal(10) ** 5,
    "crore": Decimal(10) ** 7,
    "cr": Decimal(10) ** 7,
    "million": Decimal(10) ** 6,
    "thousand": Decimal(10) ** 3,
    "k": Decimal(10) ** 3,
}

_CURRENCY_NOISE = ("\u20b9", "rs.", "rs", "inr", ",", "_")


def coerce_decimal(value: Any) -> Decimal | None:
    """Best-effort deterministic numeric coercion ('81,000', '2.5 lakh' -> Decimal)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    for noise in _CURRENCY_NOISE:
        text = text.replace(noise, "")
    text = text.strip()
    if text.endswith("+"):
        text = text[:-1].strip()
    multiplier: Decimal | None = None
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if text.endswith(suffix):
            multiplier = _SUFFIXES[suffix]
            text = text[: -len(suffix)].strip()
            break
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return number * multiplier if multiplier is not None else number


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


# --------------------------------------------------------------------------- outcome


@dataclass
class OperatorOutcome:
    passed: bool | None
    comparison: str = ""
    threshold: Any | None = None
    threshold_variant: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    note: str | None = None
    extra_missing: list[str] = field(default_factory=list)


OperatorFn = Callable[[Any, RuleSpec, dict[str, Any]], OperatorOutcome]

_DIRECTION_SYMBOL = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "=="}


def _compare(actual: Decimal, threshold: Decimal, direction: str) -> bool:
    if direction == "lt":
        return actual < threshold
    if direction == "lte":
        return actual <= threshold
    if direction == "gt":
        return actual > threshold
    if direction == "gte":
        return actual >= threshold
    if direction == "eq":
        return actual == threshold
    raise KeyError(direction)


def _numeric_outcome(
    actual: Any, threshold_value: Any, direction: str, *, variant: str | None = None
) -> OperatorOutcome:
    actual_dec = coerce_decimal(actual)
    threshold_dec = coerce_decimal(threshold_value)
    if actual_dec is None or threshold_dec is None:
        return OperatorOutcome(
            passed=None,
            note="unparseable numeric value",
            missing_fields=[] if actual_dec is not None else [""],
        )
    passed = _compare(actual_dec, threshold_dec, direction)
    comparison = f"{actual_dec} {_DIRECTION_SYMBOL.get(direction, direction)} {threshold_dec}"
    if variant:
        comparison += f" [{variant}]"
    return OperatorOutcome(
        passed=passed,
        comparison=comparison,
        threshold=threshold_value,
        threshold_variant=variant,
    )


# ------------------------------------------------------------------------- operators


def op_eq(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    expected = spec.value
    if coerce_decimal(actual) is not None and coerce_decimal(expected) is not None:
        return _numeric_outcome(actual, expected, "eq")
    return OperatorOutcome(
        passed=normalize_text(actual) == normalize_text(expected),
        comparison=f"{normalize_text(actual)!r} == {normalize_text(expected)!r}",
        threshold=expected,
    )


def op_neq(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    outcome = op_eq(actual, spec, facts)
    return OperatorOutcome(
        passed=(not outcome.passed) if outcome.passed is not None else None,
        comparison=f"{outcome.comparison} (negated)",
        threshold=outcome.threshold,
    )


def _direction_op(direction: str) -> OperatorFn:
    def run(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
        return _numeric_outcome(actual, spec.value, direction)

    return run


op_gt = _direction_op("gt")
op_gte = _direction_op("gte")
op_lt = _direction_op("lt")
op_lte = _direction_op("lte")


def op_between(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    low = coerce_decimal(spec.min_value)
    high = coerce_decimal(spec.max_value)
    actual_dec = coerce_decimal(actual)
    if actual_dec is None:
        return OperatorOutcome(passed=None, note="unparseable numeric value")
    parts = []
    passed = True
    if low is not None:
        passed = passed and actual_dec >= low
        parts.append(f">= {low}")
    if high is not None:
        passed = passed and actual_dec <= high
        parts.append(f"<= {high}")
    if not parts:
        return OperatorOutcome(passed=None, note="between requires min_value and/or max_value")
    return OperatorOutcome(
        passed=passed,
        comparison=f"{actual_dec} in {' and '.join(parts)}",
        threshold={"min": spec.min_value, "max": spec.max_value},
    )


def _membership(actual: Any, spec: RuleSpec) -> tuple[bool | None, str]:
    values = spec.values or []
    if not values:
        return None, "in/not_in requires values[]"
    actual_dec = coerce_decimal(actual)
    if actual_dec is not None:
        for candidate in values:
            candidate_dec = coerce_decimal(candidate)
            if candidate_dec is not None and candidate_dec == actual_dec:
                return True, f"{actual_dec} in {values}"
    return normalize_text(actual) in {normalize_text(v) for v in values}, (
        f"{normalize_text(actual)!r} in {[str(v) for v in values]}"
    )


def op_in(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    passed, comparison = _membership(actual, spec)
    if passed is None:
        return OperatorOutcome(passed=None, note=comparison)
    return OperatorOutcome(passed=passed, comparison=comparison, threshold=spec.values)


def op_not_in(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    passed, comparison = _membership(actual, spec)
    if passed is None:
        return OperatorOutcome(passed=None, note=comparison)
    return OperatorOutcome(passed=not passed, comparison=f"NOT ({comparison})", threshold=spec.values)


def op_exists(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    return OperatorOutcome(passed=not is_blank(actual), comparison=f"field present: {not is_blank(actual)}")


def op_threshold_map(actual: Any, spec: RuleSpec, facts: dict[str, Any]) -> OperatorOutcome:
    """Location-dependent threshold lookup (e.g. rural vs urban income limits).

    The threshold is selected by the value of `location_field` in the facts:
      * exact variant match (rural/urban) wins;
      * else "default" if provided;
      * else a single-entry map is used as-is;
      * otherwise the rule reports missing_data for the location field.
    """
    thresholds = spec.thresholds or {}
    direction = spec.direction or "lte"
    location_field = spec.location_field or "location_type"
    location_value = facts.get(location_field)

    variant: str | None = None
    threshold_value: Any | None = None
    if not is_blank(location_value) and normalize_text(location_value) in {
        normalize_text(k) for k in thresholds
    }:
        for key, val in thresholds.items():
            if normalize_text(key) == normalize_text(location_value):
                variant, threshold_value = str(key), val
                break
    elif "default" in thresholds:
        variant, threshold_value = "default", thresholds["default"]
    elif len(thresholds) == 1:
        variant, threshold_value = next(iter(thresholds.items()))
    else:
        return OperatorOutcome(
            passed=None,
            note=f"no threshold variant for {location_field}={location_value!r}",
            missing_fields=[location_field],
        )

    outcome = _numeric_outcome(actual, threshold_value, direction, variant=variant)
    if outcome.passed is None and outcome.missing_fields == [""]:
        # unparseable actual value -> attribute it to the rule's field
        outcome.missing_fields = []
        outcome.note = "unparseable numeric value"
    return outcome


OPERATORS: dict[str, OperatorFn] = {
    "eq": op_eq,
    "neq": op_neq,
    "gt": op_gt,
    "gte": op_gte,
    "lt": op_lt,
    "lte": op_lte,
    "between": op_between,
    "in": op_in,
    "not_in": op_not_in,
    "exists": op_exists,
    "threshold_map": op_threshold_map,
}


def display_value(value: Any, unit: str | None) -> str | None:
    if value is None:
        return None
    if unit == "INR":
        decimal_value = coerce_decimal(value)
        if decimal_value is not None:
            return format_inr(decimal_value)
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)
