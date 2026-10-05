"""Financial engine tests - Decimal precision, invalid inputs, edge tenures, determinism."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.services.financials import (
    FinancialEngineError,
    amortization_schedule,
    calculate_emi,
    make_terms,
    quote,
    resolve_terms,
)


def test_emi_known_value():
    # 1,00,000 @ 10% p.a. for 12 months -> 8,791.59 (standard reducing-balance EMI)
    emi = calculate_emi(Decimal("100000"), Decimal("10"), 12)
    assert emi == Decimal("8791.59")
    assert isinstance(emi, Decimal)


def test_emi_is_quantized_to_two_decimals():
    emi = calculate_emi(Decimal("99999.99"), Decimal("7.25"), 7)
    exponent = emi.as_tuple().exponent
    assert exponent == -2, f"expected 2dp money, got {emi}"


def test_emi_deterministic_same_input_same_output():
    args = (Decimal("543210.55"), Decimal("8.75"), 29)
    assert calculate_emi(*args) == calculate_emi(*args)


def test_zero_interest_loan_splits_principal_evenly():
    assert calculate_emi(Decimal("12000"), Decimal("0"), 12) == Decimal("1000.00")


def test_one_month_tenure():
    # single installment: principal + one month of interest
    assert calculate_emi(Decimal("10000"), Decimal("12"), 1) == Decimal("10100.00")


def test_long_tenure_360_months():
    emi = calculate_emi(Decimal("5000000"), Decimal("8.5"), 360)
    assert emi > 0
    schedule = amortization_schedule(make_terms(Decimal("5000000"), Decimal("8.5"), 360))
    assert len(schedule) == 360
    assert schedule[-1].closing_balance == Decimal("0.00")
    # sum of principal components repays the principal exactly (rounding absorbed in final row)
    total_principal = sum((row.principal_component for row in schedule), Decimal("0"))
    assert abs(total_principal - Decimal("5000000")) <= Decimal("1.00")


@pytest.mark.parametrize(
    "principal,rate,tenure,code",
    [
        (0, 10, 12, "invalid_principal"),
        (-5, 10, 12, "invalid_principal"),
        (100000, -1, 12, "invalid_rate"),
        (100000, 101, 12, "invalid_rate"),
        (100000, 10, 0, "invalid_tenure"),
        (100000, 10, -3, "invalid_tenure"),
        (100000, 10, 601, "invalid_tenure"),
        (100000, 10, 12.5, "invalid_tenure"),
        (100000, 10, True, "invalid_tenure"),
        ("not-a-number", 10, 12, "invalid_value"),
        (100000, None, 12, "invalid_value"),
    ],
)
def test_invalid_inputs_raise_typed_errors(principal, rate, tenure, code):
    with pytest.raises(FinancialEngineError) as excinfo:
        calculate_emi(principal, rate, tenure)
    assert excinfo.value.code == code


def test_quote_totals_are_consistent():
    terms = make_terms(Decimal("250000"), Decimal("12.5"), 36)
    result = quote(terms, processing_fee_percent=Decimal("1.5"))
    schedule = amortization_schedule(terms)
    assert result.total_payable == sum((row.emi for row in schedule), Decimal("0"))
    assert result.total_interest == result.total_payable - result.principal
    assert result.processing_fee == Decimal("3750.00")
    assert result.net_disbursed == Decimal("246250.00")
    assert result.total_cost_of_credit == result.total_interest + result.processing_fee


def test_quote_zero_fee():
    result = quote(make_terms(Decimal("60000"), Decimal("12"), 24), processing_fee_percent=0)
    assert result.processing_fee == Decimal("0.00")
    assert result.net_disbursed == Decimal("60000.00")


def test_quote_rejects_absurd_processing_fee():
    with pytest.raises(FinancialEngineError) as excinfo:
        quote(make_terms(Decimal("60000"), Decimal("12"), 24), processing_fee_percent=Decimal("99"))
    assert excinfo.value.code == "invalid_processing_fee"


def test_schedule_final_installment_clears_balance():
    terms = make_terms(Decimal("81000"), Decimal("10"), 12)
    schedule = amortization_schedule(terms)
    assert schedule[-1].closing_balance == Decimal("0.00")
    assert schedule[0].opening_balance == Decimal("81000.00")
    assert all(row.emi > 0 for row in schedule)


def test_schedule_chains_balances():
    schedule = amortization_schedule(make_terms(Decimal("81000"), Decimal("10"), 12))
    for previous, current in zip(schedule, schedule[1:]):
        assert current.opening_balance == previous.closing_balance


# ------------------------------------------------------------------ resolve_terms


options_kwargs = dict(
    min_principal=Decimal("5000"),
    max_principal=Decimal("60000"),
    annual_interest_rate_percent=Decimal("12"),
    tenure_options=[12, 18, 24, 30, 36],
    default_tenure_months=24,
)


def test_resolve_terms_defaults_to_max_principal():
    terms = resolve_terms(**options_kwargs)
    assert terms.principal == Decimal("60000")
    assert terms.tenure_months == 24


def test_resolve_terms_clamps_requested_principal():
    over = resolve_terms(**options_kwargs, requested_principal=Decimal("999999"))
    under = resolve_terms(**options_kwargs, requested_principal=Decimal("100"))
    assert over.principal == Decimal("60000")
    assert under.principal == Decimal("5000")


def test_resolve_terms_tenure_snapping():
    exact = resolve_terms(**options_kwargs, requested_tenure_months=30)
    between = resolve_terms(**options_kwargs, requested_tenure_months=28)  # -> largest option <= 28
    too_small = resolve_terms(**options_kwargs, requested_tenure_months=6)  # -> smallest option
    assert exact.tenure_months == 30
    assert between.tenure_months == 24
    assert too_small.tenure_months == 12


def test_resolve_terms_ignores_garbage_requests():
    terms = resolve_terms(**options_kwargs, requested_principal="abc", requested_tenure_months="long")
    assert terms.principal == Decimal("60000")
    assert terms.tenure_months == 24
