"""Loads/validates scheme rule data from data/scheme_rules/*.json."""

from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.rules.operators import OPERATORS
from app.schemas.rules import RuleSpec


class RuleDataError(RuntimeError):
    """Raised when a scheme rule file is missing or malformed."""


class SchemeSource(BaseModel):
    type: str = "demo_placeholder"
    url: str | None = None
    note: str | None = None
    document_file: str | None = None


class SchemeFinancials(BaseModel):
    min_principal: Decimal = Decimal("0")
    max_principal: Decimal
    interest_rate_annual_percent: Decimal
    tenure_months_options: list[int]
    default_tenure_months: int
    processing_fee_percent: Decimal = Decimal("0")
    moratorium_months: int | None = None
    repayment_frequency: str = "monthly"
    source_note: str | None = None

    @model_validator(mode="after")
    def _check_tenure(self) -> "SchemeFinancials":
        if not self.tenure_months_options:
            raise ValueError("tenure_months_options must not be empty")
        if self.default_tenure_months not in self.tenure_months_options:
            raise ValueError("default_tenure_months must be one of tenure_months_options")
        if self.max_principal <= 0:
            raise ValueError("max_principal must be > 0")
        if self.min_principal < 0:
            raise ValueError("min_principal must be >= 0")
        return self


class SchemeRuleSet(BaseModel):
    scheme_id: str
    name: str
    version: int = 1
    category: str | None = None
    description: str = ""
    demo_data: bool = True
    tags: list[str] = Field(default_factory=list)
    source: SchemeSource = SchemeSource()
    eligibility_rules: list[RuleSpec] = Field(default_factory=list)
    financials: SchemeFinancials
    claims: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_rules(self) -> "SchemeRuleSet":
        seen: set[str] = set()
        for rule in self.eligibility_rules:
            if rule.rule_id in seen:
                raise ValueError(f"duplicate rule_id {rule.rule_id!r} in {self.scheme_id}")
            seen.add(rule.rule_id)
            if rule.operator not in OPERATORS:
                raise ValueError(f"unknown operator {rule.operator!r} in rule {rule.rule_id!r}")
        return self


class RuleRepository:
    def __init__(self, rules_dir: Path):
        self.rules_dir = rules_dir
        self._schemes: dict[str, SchemeRuleSet] = {}
        self.load_all()

    def load_all(self) -> dict[str, SchemeRuleSet]:
        schemes: dict[str, SchemeRuleSet] = {}
        if not self.rules_dir.exists():
            raise RuleDataError(f"rules directory not found: {self.rules_dir}")
        for path in sorted(self.rules_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                scheme = SchemeRuleSet.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 - surfaced as RuleDataError with file context
                raise RuleDataError(f"invalid rule file {path.name}: {exc}") from exc
            if scheme.scheme_id in schemes:
                raise RuleDataError(f"duplicate scheme_id {scheme.scheme_id!r} ({path.name})")
            schemes[scheme.scheme_id] = scheme
        self._schemes = schemes
        return schemes

    def all(self) -> list[SchemeRuleSet]:
        return list(self._schemes.values())

    def get(self, scheme_id: str) -> SchemeRuleSet | None:
        return self._schemes.get(scheme_id)

    def ids(self) -> list[str]:
        return list(self._schemes.keys())


@lru_cache
def get_rule_repository() -> RuleRepository:
    return RuleRepository(get_settings().scheme_rules_dir)
