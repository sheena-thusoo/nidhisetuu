"""Intake agent: extract structured facts from free text.

Strategy (merged deterministically):
  1. regex extractor  - high precision for numbers/amounts/categories; ALWAYS runs;
  2. Groq LLM extractor - fills fields regex missed (live mode only).
Explicit API-provided values always win over extracted ones.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

from app.ai.json_utils import extract_json
from app.ai.llm import LLMClient, LLMError, get_llm_client
from app.rules.operators import coerce_decimal
from app.schemas.intake import ExtractedField, IntakeExtraction
from app.utils.logging import log_event, log_mock

logger = logging.getLogger(__name__)

INDIAN_STATES = {
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "delhi": "Delhi",
    "jammu and kashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "puducherry": "Puducherry",
    "chandigarh": "Chandigarh",
}

_AMOUNT = r"((?:\d[\d,]*(?:\.\d+)?)\s*(?:lakhs?|lacs?|crores?|cr|k|thousand)?)"
KNOWLEDGE_FIELDS = (
    "annual_income",
    "category",
    "location_type",
    "age",
    "course_level",
    "loan_amount_requested",
)


def _search_amount(pattern: str, text: str) -> Decimal | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return coerce_decimal(match.group(1))


def _extract_category(text: str) -> str | None:
    match = re.search(r"\b(sc|st|obc|ebc|dnt|general)\b", text, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).upper()
    return value if value != "GENERAL" else "General"


def _extract_location(text: str) -> str | None:
    if re.search(r"\b(rural|village|villager|gramin|gaon)\b", text, re.IGNORECASE):
        return "rural"
    if re.search(r"\b(urban|city|town|municipal|metro|metropolitan)\b", text, re.IGNORECASE):
        return "urban"
    return None


def _extract_course(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(phd|doctorate|postgraduate|post graduate|master'?s?|m\.?tech|mba|m\.?sc|m\.?com|ma)\b", lowered):
        return "postgraduate"
    if re.search(r"\b(engineering|medical|mbbs|professional|b\.?tech|ca\b|law)\b", lowered):
        return "professional"
    if re.search(r"\b(diploma|polytechnic|iti)\b", lowered):
        return "diploma"
    if re.search(r"\b(certificate|certification|skill course)\b", lowered):
        return "certificate"
    if re.search(r"\b(undergraduate|under graduate|bachelor|graduation|degree|b\.?sc|b\.?com|ba\b)\b", lowered):
        return "undergraduate"
    return None


def _extract_age(text: str) -> int | None:
    patterns = (
        r"\bage\D{0,12}?(\d{1,2})\b",
        r"\b(\d{1,2})\s*(?:years?|yrs?)[\s-]*(?:old|of age)",
        r"\bi am\s*(\d{1,2})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return age
    return None


def _extract_gender(text: str) -> str | None:
    if re.search(r"\b(transgender|trans woman|trans man)\b", text, re.IGNORECASE):
        return "transgender"
    if re.search(r"\b(female|woman|women|mahila|girl)\b", text, re.IGNORECASE):
        return "female"
    if re.search(r"\b(male|man|men|boy)\b", text, re.IGNORECASE):
        return "male"
    return None


def _extract_state(text: str) -> str | None:
    lowered = text.lower()
    for key, value in INDIAN_STATES.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return value
    return None


def _extract_existing_loan(text: str) -> bool | None:
    lowered = text.lower()
    if re.search(r"\b(no|not|never|without|clear of)\b[^.]{0,25}\b(default|existing loan|loan)\b", lowered):
        return False
    if re.search(r"\b(existing loan|loan default|defaulted|defaulter|npa|overdue)\b", lowered):
        return True
    return None


def _extract_business_vintage(text: str) -> Decimal | None:
    pattern = (
        r"\b(?:running|operating|in business|business|enterprise|shop|working|established)\b"
        r"\D{0,30}?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return coerce_decimal(match.group(1))


def _extract_tenure(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(?:months?|mahine)\b", text, re.IGNORECASE)
    if match:
        tenure = int(match.group(1))
        if 1 <= tenure <= 600:
            return tenure
    return None


def extract_intake_from_text(text: str) -> dict[str, Any]:
    """Deterministic regex extraction. Runs in BOTH live and mock modes."""
    facts: dict[str, Any] = {}

    monthly = _search_amount(r"(?:monthly|per month|har month)\s+(?:family\s+|household\s+)?income\D{0,15}?" + _AMOUNT, text)
    if monthly is not None:
        facts["annual_income"] = monthly * 12
    else:
        income = _search_amount(
            r"(?:annual|yearly|family|household|total)?\s*income\D{0,40}?" + _AMOUNT, text
        )
        if income is not None:
            facts["annual_income"] = income

    loan_amount = _search_amount(
        r"\b(?:loan|amount|tuition|fees?|need(?:ed)?|require(?:d)?|cost)\b\D{0,30}?" + _AMOUNT, text
    )
    if loan_amount is not None:
        facts["loan_amount_requested"] = loan_amount

    category = _extract_category(text)
    if category:
        facts["category"] = category
    location = _extract_location(text)
    if location:
        facts["location_type"] = location
    age = _extract_age(text)
    if age is not None:
        facts["age"] = age
    course = _extract_course(text)
    if course:
        facts["course_level"] = course
    gender = _extract_gender(text)
    if gender:
        facts["gender"] = gender
    state = _extract_state(text)
    if state:
        facts["state"] = state
    vintage = _extract_business_vintage(text)
    if vintage is not None:
        facts["years_in_operation"] = vintage
    tenure = _extract_tenure(text)
    if tenure is not None:
        facts["tenure_months"] = tenure
    existing = _extract_existing_loan(text)
    if existing is not None:
        facts["existing_loan"] = existing
    return facts


INTAKE_SYSTEM_PROMPT = (
    "You extract structured applicant facts for Indian government loan/education schemes. "
    "Return ONLY a JSON object. Allowed keys: annual_income (number, INR per year), category "
    "(SC|ST|OBC|EBC|DNT|General), location_type (rural|urban), state, age (integer), gender, "
    "course_level (undergraduate|postgraduate|professional|diploma|certificate), "
    "years_in_operation (number), existing_loan (boolean), loan_amount_requested (number INR), "
    "tenure_months (integer). Omit keys you cannot infer. Never guess numbers. "
    "Never compute eligibility, EMI or interest."
)
_ALLOWED_LLM_KEYS = {
    "annual_income",
    "category",
    "location_type",
    "state",
    "age",
    "gender",
    "course_level",
    "years_in_operation",
    "existing_loan",
    "loan_amount_requested",
    "tenure_months",
}


async def run_intake(
    text: str | None,
    provided: dict[str, Any] | None = None,
    *,
    llm: LLMClient | None = None,
    use_live_llm: bool = True,
) -> IntakeExtraction:
    provided = {k: v for k, v in (provided or {}).items() if v is not None}
    extracted = IntakeExtraction(facts={}, provider="regex", mock_mode=llm is None)

    regex_facts: dict[str, Any] = extract_intake_from_text(text or "")
    for field, value in regex_facts.items():
        extracted.extracted.append(ExtractedField(field=field, value=value, source="regex", confidence=0.8))

    client = llm if llm is not None else (get_llm_client() if use_live_llm else None)
    if client is not None and text:
        try:
            result = await client.complete(
                INTAKE_SYSTEM_PROMPT,
                f"Applicant description:\n{text}\n\nKnown values (do not contradict): {provided or {}}",
                json_mode=True,
            )
            parsed = extract_json(result.text)
            if isinstance(parsed, dict):
                extracted.provider = f"regex+{result.provider}"
                for field, value in parsed.items():
                    if field not in _ALLOWED_LLM_KEYS or value in (None, "", []):
                        continue
                    if field in regex_facts or field in provided:
                        continue  # regex / explicit request values win
                    coerced = _coerce_llm_value(field, value)
                    if coerced is None:
                        continue
                    extracted.extracted.append(
                        ExtractedField(field=field, value=coerced, source="llm", confidence=0.6)
                    )
        except LLMError as exc:
            extracted.warnings.append(f"LLM intake failed, regex extraction kept: {exc}")
    elif text and client is None:
        log_mock(logger, "intake", "intake agent ran in regex-only mode")

    merged: dict[str, Any] = {}
    for item in extracted.extracted:
        merged[item.field] = item.value
    merged.update(provided)  # explicit API values always win
    extracted.facts = merged

    for field in KNOWLEDGE_FIELDS:
        if field not in merged:
            extracted.unresolved_questions.append(
                {
                    "annual_income": "What is the applicant's annual family income (INR)?",
                    "category": "Which category does the applicant belong to (SC/ST/OBC/EBC/DNT/General)?",
                    "location_type": "Is the applicant's location rural or urban?",
                    "age": "What is the applicant's age?",
                    "course_level": "What course level is being pursued?",
                    "loan_amount_requested": "How much loan (INR) is required?",
                }[field]
            )
    log_event(
        logger,
        logging.INFO,
        "intake extraction complete",
        fields=list(merged.keys()),
        provider=extracted.provider,
        mock_mode=extracted.mock_mode,
    )
    return extracted


def _coerce_llm_value(field: str, value: Any) -> Any | None:
    try:
        if field in {"annual_income", "loan_amount_requested", "years_in_operation"}:
            return coerce_decimal(value)
        if field in {"age", "tenure_months"}:
            number = coerce_decimal(value)
            return int(number) if number is not None else None
        if field == "existing_loan":
            return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "yes", "1"}
        if field == "category":
            text = str(value).strip().upper()
            return text if text in {"SC", "ST", "OBC", "EBC", "DNT"} else "General"
        if field == "location_type":
            text = str(value).strip().lower()
            return text if text in {"rural", "urban"} else None
        return str(value).strip() or None
    except (TypeError, ValueError):
        return None
