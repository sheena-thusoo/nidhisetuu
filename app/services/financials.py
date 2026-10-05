"""Pure financial engine - EMI / repayment math.

HARD RULE: this module imports NOTHING from app.ai, LangChain or any LLM. It is pure,
deterministic Decimal math: identical inputs always produce identical outputs.

Rounding policy: every money value is quantized to 2 decimal places with ROUND_HALF_UP
at the point where it is produced (EMI, each installment line, fees). Intermediate
compounding is done at 34-digit precision so only the final presentation rounds.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from pydantic import BaseModel

from app.utils.format import MONEY_QUANT

MAX_PRINCIPAL = Decimal("1000000000")  # 100 crore sanity ceiling
MAX_TENURE_MONTHS = 600  # 50 years
MAX_ANNUAL_RATE_PERCENT = Decimal("100")


class FinancialEngineError(ValueError):
    """Deterministic, machine-checkable validation failure from the financial engine."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class LoanTerms(BaseModel):
    principal: Decimal
    annual_interest_rate_percent: Decimal
    tenure_months: int


class Installment(BaseModel):
    month: int
    opening_balance: Decimal
    emi: Decimal
    interest_component: Decimal
    principal_component: Decimal
    closing_balance: Decimal


class LoanQuote(BaseModel):
    principal: Decimal
    annual_interest_rate_percent: Decimal
    monthly_rate_percent: Decimal
    tenure_months: int
    emi: Decimal
    total_payable: Decimal
    total_interest: Decimal
    processing_fee: Decimal
    total_cost_of_credit: Decimal
    net_disbursed: Decimal
    engine: str = "app/services/financials.py (pure deterministic engine, no LLM)"
    deterministic: bool = True
    mock_mode: bool = False


def _as_decimal(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise FinancialEngineError("invalid_value", f"{field_name} must be a number, got {value!r}")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise FinancialEngineError("invalid_value", f"{field_name} must be a number, got {value!r}") from exc


def make_terms(principal: Any, annual_interest_rate_percent: Any, tenure_months: Any) -> LoanTerms:
    principal_dec = _as_decimal(principal, "principal")
    rate_dec = _as_decimal(annual_interest_rate_percent, "annual_interest_rate_percent")

    if isinstance(tenure_months, bool) or not isinstance(tenure_months, int):
        raise FinancialEngineError("invalid_tenure", f"tenure_months must be an integer, got {tenure_months!r}")
    if principal_dec <= 0:
        raise FinancialEngineError("invalid_principal", "principal must be greater than 0")
    if principal_dec > MAX_PRINCIPAL:
        raise FinancialEngineError("invalid_principal", f"principal exceeds {MAX_PRINCIPAL}")
    if rate_dec < 0 or rate_dec > MAX_ANNUAL_RATE_PERCENT:
        raise FinancialEngineError("invalid_rate", "annual interest rate must be between 0 and 100 percent")
    if tenure_months < 1 or tenure_months > MAX_TENURE_MONTHS:
        raise FinancialEngineError("invalid_tenure", f"tenure_months must be between 1 and {MAX_TENURE_MONTHS}")
    return LoanTerms(
        principal=principal_dec, annual_interest_rate_percent=rate_dec, tenure_months=tenure_months
    )


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANT, rounding="ROUND_HALF_UP")


def calculate_emi(principal: Decimal, annual_interest_rate_percent: Decimal, tenure_months: int) -> Decimal:
    """Standard reducing-balance EMI. Zero-interest loans split principal equally."""
    terms = make_terms(principal, annual_interest_rate_percent, tenure_months)
    with localcontext() as ctx:
        ctx.prec = 34
        if terms.annual_interest_rate_percent == 0:
            emi = terms.principal / Decimal(terms.tenure_months)
        else:
            monthly_rate = terms.annual_interest_rate_percent / Decimal(12) / Decimal(100)
            growth = (Decimal(1) + monthly_rate) ** terms.tenure_months
            emi = terms.principal * monthly_rate * growth / (growth - Decimal(1))
    return _quantize(emi)


def amortization_schedule(terms: LoanTerms) -> list[Installment]:
    """Month-by-month schedule; the final installment clears any rounding residue."""
    monthly_rate = terms.annual_interest_rate_percent / Decimal(12) / Decimal(100)
    emi = calculate_emi(terms.principal, terms.annual_interest_rate_percent, terms.tenure_months)
    balance = terms.principal
    rows: list[Installment] = []
    for month in range(1, terms.tenure_months + 1):
        opening = balance
        interest = _quantize(opening * monthly_rate)
        if month == terms.tenure_months:
            principal_component = opening
            emi_row = _quantize(principal_component + interest)
        else:
            principal_component = emi - interest
            if principal_component > opening:  # cannot pay more principal than outstanding
                principal_component = opening
            emi_row = emi
        closing = opening - principal_component
        rows.append(
            Installment(
                month=month,
                opening_balance=_quantize(opening),
                emi=_quantize(emi_row),
                interest_component=interest,
                principal_component=_quantize(principal_component),
                closing_balance=_quantize(closing),
            )
        )
        balance = closing
    return rows


def quote(
    terms: LoanTerms,
    *,
    processing_fee_percent: Decimal | int | float | str = Decimal("0"),
    include_schedule: bool = False,
) -> LoanQuote:
    fee_percent = _as_decimal(processing_fee_percent, "processing_fee_percent")
    if fee_percent < 0 or fee_percent > 10:
        raise FinancialEngineError("invalid_processing_fee", "processing_fee_percent must be between 0 and 10")

    monthly_rate = terms.annual_interest_rate_percent / Decimal(12) / Decimal(100)
    emi = calculate_emi(terms.principal, terms.annual_interest_rate_percent, terms.tenure_months)
    schedule = amortization_schedule(terms)
    total_payable = sum((row.emi for row in schedule), Decimal("0"))
    total_interest = total_payable - terms.principal
    processing_fee = _quantize(terms.principal * fee_percent / Decimal(100))

    result = LoanQuote(
        principal=_quantize(terms.principal),
        annual_interest_rate_percent=terms.annual_interest_rate_percent,
        monthly_rate_percent=monthly_rate,
        tenure_months=terms.tenure_months,
        emi=emi,
        total_payable=_quantize(total_payable),
        total_interest=_quantize(total_interest),
        processing_fee=processing_fee,
        total_cost_of_credit=_quantize(total_interest + processing_fee),
        net_disbursed=_quantize(terms.principal - processing_fee),
    )
    if include_schedule:
        result.__dict__["schedule"] = schedule  # attached only for callers that ask
    return result


def resolve_terms(
    min_principal: Decimal,
    max_principal: Decimal,
    annual_interest_rate_percent: Decimal,
    tenure_options: list[int],
    default_tenure_months: int,
    *,
    requested_principal: Any | None = None,
    requested_tenure_months: Any | None = None,
) -> LoanTerms:
    """Deterministically clamp a user request to the scheme's configured limits."""
    if requested_principal is None:
        principal = max_principal
    else:
        try:
            requested = _as_decimal(requested_principal, "requested_principal")
        except FinancialEngineError:
            requested = max_principal
        if requested <= 0:
            principal = max_principal
        else:
            principal = max(min(requested, max_principal), min_principal if min_principal > 0 else Decimal("0.01"))
    principal = min(principal, max_principal)

    options = sorted(tenure_options)
    tenure = default_tenure_months
    if isinstance(requested_tenure_months, int) and not isinstance(requested_tenure_months, bool):
        if requested_tenure_months in options:
            tenure = requested_tenure_months
        else:
            smaller = [o for o in options if o <= requested_tenure_months]
            tenure = max(smaller) if smaller else min(options)
    return make_terms(principal, annual_interest_rate_percent, tenure)
