"""Intake agent tests: regex extraction, merge precedence, unresolved questions, numeric guard.

Mock mode is forced by tests/conftest.py (GROQ_API_KEY=""), so run_intake always uses the
deterministic regex path unless an explicit fake LLM is injected.
"""

from __future__ import annotations

from decimal import Decimal

from app.ai.intake import extract_intake_from_text, run_intake
from app.ai.explanation import (
    ExplanationResult,
    collect_allowed_numbers,
    generate_explanation,
    numbers_in_text,
    numeric_guard,
    template_explanation,
)


class _FakeLLM:
    """Minimal LLMClient double: returns canned JSON/text and records calls."""

    provider = "fake-groq"
    model = "fake-model"
    mock_mode = False

    def __init__(self, text: str):
        self.text = text
        self.calls: list[tuple[str, str, bool]] = []

    async def complete(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0):
        self.calls.append((system, user, json_mode))
        from app.ai.llm import LLMResult

        return LLMResult(text=self.text, provider=self.provider, model=self.model)


# ------------------------------------------------------------------ regex extraction


def test_income_and_loan_amount_extraction():
    facts = extract_intake_from_text(
        "My family income is 81,000 per year and I need a loan of 1.5 lakh for my diploma."
    )
    assert facts["annual_income"] == Decimal("81000")
    assert facts["loan_amount_requested"] == Decimal("150000")


def test_monthly_income_is_multiplied_by_twelve():
    facts = extract_intake_from_text("monthly family income is 7000")
    assert facts["annual_income"] == Decimal("84000")


def test_lakh_and_crore_suffixes_scale_the_amount():
    assert extract_intake_from_text("income 2.5 lakh per year")["annual_income"] == Decimal("250000")
    assert extract_intake_from_text("need loan of 1 crore rupees")["loan_amount_requested"] == Decimal(
        10**7
    )


def test_category_location_and_age():
    facts = extract_intake_from_text(
        "I am a 19 year old rural SC student living in a village in Bihar."
    )
    assert facts["category"] == "SC"
    assert facts["location_type"] == "rural"
    assert facts["age"] == 19
    assert facts["state"] == "Bihar"


def test_urban_location_and_course_level():
    facts = extract_intake_from_text("I live in a city and am doing a btech engineering course")
    assert facts["location_type"] == "urban"
    assert facts["course_level"] == "professional"


def test_gender_extraction():
    assert extract_intake_from_text("I am a woman running a shop")["gender"] == "female"
    assert extract_intake_from_text("male applicant")["gender"] == "male"


def test_business_vintage_and_existing_loan():
    facts = extract_intake_from_text("I have been running a tailoring business 3 years with no existing loan")
    assert facts["years_in_operation"] == Decimal("3")
    assert facts["existing_loan"] is False


def test_tenure_months_extraction():
    assert extract_intake_from_text("repay in 24 months")["tenure_months"] == 24


def test_nothing_extracted_from_empty_text():
    assert extract_intake_from_text("") == {}


# ------------------------------------------------------------- merge + intake output


async def test_provided_values_win_over_regex():
    extraction = await run_intake(
        "family income is 50000 per year",
        provided={"annual_income": "65000", "category": "OBC"},
        use_live_llm=False,
    )
    assert extraction.facts["annual_income"] == "65000"  # provided values pass through as-is
    assert extraction.facts["category"] == "OBC"  # rule operators coerce/compare later
    assert extraction.mock_mode is True
    assert extraction.provider == "regex"


async def test_llm_fills_fields_regex_missed():
    llm = _FakeLLM('{"course_level": "postgraduate", "state": "Kerala"}')
    extraction = await run_intake(
        "My family income is 90000 yearly.", provided={}, llm=llm, use_live_llm=True
    )
    assert extraction.facts["annual_income"] == Decimal("90000")  # regex
    assert extraction.facts["course_level"] == "postgraduate"  # LLM fill
    assert extraction.facts["state"] == "Kerala"  # LLM fill
    assert extraction.provider == "regex+fake-groq"
    assert extraction.mock_mode is False


async def test_llm_cannot_override_regex_or_provided():
    llm = _FakeLLM('{"annual_income": 999999, "category": "General"}')
    extraction = await run_intake(
        "family income is 40000 per year, SC applicant",
        provided={"category": "ST"},
        llm=llm,
        use_live_llm=True,
    )
    assert extraction.facts["annual_income"] == Decimal("40000")  # regex beats LLM
    assert extraction.facts["category"] == "ST"  # provided beats LLM


async def test_llm_disallowed_fields_are_dropped():
    llm = _FakeLLM('{"eligibility": "approved", "emi": 1234, "course_level": "diploma"}')
    extraction = await run_intake("student", llm=llm, use_live_llm=True)
    assert "eligibility" not in extraction.facts
    assert "emi" not in extraction.facts
    assert extraction.facts["course_level"] == "diploma"


async def test_llm_failure_keeps_regex_facts():
    from app.ai.llm import LLMError

    class _BrokenLLM(_FakeLLM):
        async def complete(self, system, user, *, json_mode=False, temperature=0.0):
            raise LLMError("boom")

    extraction = await run_intake("family income is 45000 per year", llm=_BrokenLLM(""), use_live_llm=True)
    assert extraction.facts["annual_income"] == Decimal("45000")
    assert any("LLM intake failed" in w for w in extraction.warnings)


async def test_unresolved_questions_cover_knowledge_fields():
    extraction = await run_intake("hello there", provided={}, use_live_llm=False)
    assert any("income" in q.lower() for q in extraction.unresolved_questions)
    assert any("category" in q.lower() for q in extraction.unresolved_questions)
    # all six KNOWLEDGE_FIELDS produce a question
    assert len(extraction.unresolved_questions) == 6


async def test_no_unresolved_questions_when_all_fields_known():
    extraction = await run_intake(
        "rural SC student age 20 undergraduate income 70000 need 50000",
        use_live_llm=False,
    )
    assert extraction.unresolved_questions == []


# ---------------------------------------------------------------- explanation guard


def test_numbers_in_text_normalizes_commas_and_trailing_zeros():
    assert numbers_in_text("INR 1,00,000.00 and 10%") == {"100000", "10"}


def test_numeric_guard_flags_unknown_numbers():
    allowed = collect_allowed_numbers({"emi": "8791.59", "rate": 10})
    ok, offending = numeric_guard("EMI is 8791.59 at rate 10 but you could pay 9000 extra", allowed)
    assert not ok
    assert "9000" in offending


def test_numeric_guard_passes_for_pure_paraphrase():
    allowed = collect_allowed_numbers({"emi": "8791.59", "rate": 10})
    ok, offending = numeric_guard("The monthly instalment is 8791.59 at 10 percent.", allowed)
    assert ok
    assert offending == set()


def test_template_explanation_includes_status_and_disclaimer_line():
    payload = {
        "facts": {"location_type": "rural"},
        "evaluations": [
            {
                "name": "Demo Scheme",
                "eligibility": {
                    "status": "potentially_eligible",
                    "reasons": [],
                    "missing_fields": [],
                },
            }
        ],
        "verdicts": [],
    }
    text = template_explanation(payload)
    assert "Demo Scheme" in text
    assert "potentially eligible" in text
    assert "INSUFFICIENT_EVIDENCE" in text
    assert "deterministic financial engine" in text


async def test_explanation_mock_mode_uses_template():
    result = await generate_explanation(
        {"facts": {}, "evaluations": [], "verdicts": []}, use_live_llm=True
    )
    assert isinstance(result, ExplanationResult)
    assert result.provider == "template"
    assert result.mock_mode is True
    assert result.guard_triggered is False


async def test_explanation_llm_text_discarded_when_guard_violated():
    llm = _FakeLLM("You are approved! Pay only 1234.56 per month forever.")
    result = await generate_explanation(
        {"facts": {"annual_income": 70000}, "evaluations": [], "verdicts": []}, llm=llm
    )
    assert result.guard_triggered is True
    assert result.provider == "template"
    assert "1234.56" not in result.text
    assert any("numeric_guard" in w or "llm_numeric_guard" in w for w in result.warnings)


async def test_explanation_llm_text_kept_when_guard_clean():
    llm = _FakeLLM(
        "Based on the deterministic evaluation, the applicant appears potentially eligible. "
        "No figures are stated here."
    )
    result = await generate_explanation(
        {"facts": {"annual_income": 70000}, "evaluations": [], "verdicts": []}, llm=llm
    )
    assert result.guard_triggered is False
    assert result.provider == "fake-groq"
    assert result.mock_mode is False
