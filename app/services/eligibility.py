"""Eligibility orchestration: rule engine + financial engine, both deterministic.

This service is deliberately LLM-free: the agentic layer calls it (through the tool
gateway) and only interprets/explains its structured output.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from app.constants import DISCLAIMER
from app.rules.engine import evaluate_rules, summarize
from app.rules.loader import RuleRepository, SchemeRuleSet
from app.schemas.rules import EligibilityResult
from app.services.financials import quote, resolve_terms
from app.utils.format import format_inr


class FinancialConfig(BaseModel):
    """The scheme's configured (rule-data) financial parameters - not a computed quote."""

    min_principal: Decimal
    max_principal: Decimal
    interest_rate_annual_percent: Decimal
    tenure_months_options: list[int]
    default_tenure_months: int
    processing_fee_percent: Decimal = Decimal("0")
    repayment_frequency: str = "monthly"
    source_note: str | None = None


class EvaluatedFinancials(BaseModel):
    """Financial block - ALWAYS produced by app/services/financials.py, never by an LLM."""

    loan_amount: Decimal
    loan_amount_display: str | None = None
    tenure_months: int
    annual_interest_rate_percent: Decimal
    monthly_emi: Decimal
    monthly_emi_display: str | None = None
    total_interest: Decimal
    total_payable: Decimal
    processing_fee: Decimal
    total_cost_of_credit: Decimal
    net_disbursed: Decimal
    engine: str = "app/services/financials.py"
    deterministic: bool = True
    mock_mode: bool = False


class SchemeEvaluation(BaseModel):
    scheme_id: str
    name: str
    version: int
    category: str | None = None
    demo_data: bool = True
    eligibility: EligibilityResult
    financials: EvaluatedFinancials | None = None
    financial_config: FinancialConfig | None = None
    claims: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def evaluate_scheme(
    scheme: SchemeRuleSet,
    facts: dict[str, Any],
    *,
    include_financials: bool = True,
) -> SchemeEvaluation:
    rule_results = evaluate_rules(scheme.eligibility_rules, facts)
    eligibility = summarize(scheme.scheme_id, scheme.version, rule_results)

    financials: EvaluatedFinancials | None = None
    if include_financials:
        financials = build_financials(scheme, facts)

    cfg = scheme.financials
    return SchemeEvaluation(
        scheme_id=scheme.scheme_id,
        name=scheme.name,
        version=scheme.version,
        category=scheme.category,
        demo_data=scheme.demo_data,
        eligibility=eligibility,
        financials=financials,
        financial_config=FinancialConfig(
            min_principal=cfg.min_principal,
            max_principal=cfg.max_principal,
            interest_rate_annual_percent=cfg.interest_rate_annual_percent,
            tenure_months_options=cfg.tenure_months_options,
            default_tenure_months=cfg.default_tenure_months,
            processing_fee_percent=cfg.processing_fee_percent,
            repayment_frequency=cfg.repayment_frequency,
            source_note=cfg.source_note,
        ),
        claims=list(scheme.claims),
        source=scheme.source.model_dump(),
        disclaimer=DISCLAIMER,
    )


def build_financials(scheme: SchemeRuleSet, facts: dict[str, Any]) -> EvaluatedFinancials:
    cfg = scheme.financials
    terms = resolve_terms(
        cfg.min_principal,
        cfg.max_principal,
        cfg.interest_rate_annual_percent,
        cfg.tenure_months_options,
        cfg.default_tenure_months,
        requested_principal=facts.get("loan_amount_requested"),
        requested_tenure_months=facts.get("tenure_months"),
    )
    result = quote(terms, processing_fee_percent=cfg.processing_fee_percent)
    return EvaluatedFinancials(
        loan_amount=result.principal,
        loan_amount_display=format_inr(result.principal),
        tenure_months=result.tenure_months,
        annual_interest_rate_percent=result.annual_interest_rate_percent,
        monthly_emi=result.emi,
        monthly_emi_display=format_inr(result.emi),
        total_interest=result.total_interest,
        total_payable=result.total_payable,
        processing_fee=result.processing_fee,
        total_cost_of_credit=result.total_cost_of_credit,
        net_disbursed=result.net_disbursed,
    )


def match_schemes(
    repo: RuleRepository,
    facts: dict[str, Any],
    *,
    scheme_ids: list[str] | None = None,
    include_financials: bool = True,
) -> list[SchemeEvaluation]:
    schemes = [repo.get(sid) for sid in scheme_ids] if scheme_ids else repo.all()
    evaluations = [
        evaluate_scheme(scheme, facts, include_financials=include_financials)
        for scheme in schemes
        if scheme is not None
    ]
    priority = {"potentially_eligible": 0, "insufficient_data": 1, "not_eligible": 2}
    return sorted(evaluations, key=lambda e: (priority[e.eligibility.status], e.scheme_id))
