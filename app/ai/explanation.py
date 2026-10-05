"""Explanation agent - turns verified facts into human-readable text.

Numeric guard: whatever the LLM writes, every number in the final text must already
exist in the deterministic facts/evidence. A violation discards the LLM text and
falls back to the template explanation (and is reported in `warnings`).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.ai.llm import LLMClient, LLMError, get_llm_client
from app.utils.logging import log_event, log_mock

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

EXPLANATION_SYSTEM_PROMPT = (
    "You are NidhiSetu's explanation assistant for Indian government loan/education schemes. "
    "You explain results that were computed by deterministic services. Rules:\n"
    "1. NEVER invent or modify numbers. Every number you write must appear verbatim in the "
    "provided facts.\n"
    "2. Never say an applicant is approved/rejected. Use 'potentially eligible based on "
    "configured scheme criteria'.\n"
    "3. If evidence is insufficient, say so explicitly instead of guessing.\n"
    "4. Be concise: 3-6 sentences."
)


class ExplanationResult(BaseModel):
    text: str
    provider: str
    mock_mode: bool = False
    guard_triggered: bool = False
    warnings: list[str] = Field(default_factory=list)


def _normalize_number(token: str) -> str:
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


def numbers_in_text(text: str) -> set[str]:
    return {_normalize_number(match.group(0)) for match in _NUMBER_RE.finditer(text)}


def collect_allowed_numbers(payload: Any, _acc: set[str] | None = None) -> set[str]:
    """Every number that appears anywhere in the deterministic payload is 'allowed'."""
    acc = _acc if _acc is not None else set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"request_id", "trace"}:
                continue
            collect_allowed_numbers(value, acc)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            collect_allowed_numbers(item, acc)
    elif isinstance(payload, bool) or payload is None:
        pass
    elif isinstance(payload, (int, float, str)):
        for token in numbers_in_text(str(payload)):
            acc.add(token)
    return acc


def numeric_guard(text: str, allowed: set[str]) -> tuple[bool, set[str]]:
    offending = {token for token in numbers_in_text(text) if token not in allowed}
    return (not offending), offending


# ----------------------------------------------------------------------- templates


def _status_phrase(status: str) -> str:
    return {
        "potentially_eligible": "potentially eligible based on the configured scheme criteria",
        "not_eligible": "not eligible under the configured scheme criteria",
        "insufficient_data": "not yet assessable because required applicant details are missing",
    }.get(status, status)


def template_explanation(payload: dict[str, Any]) -> str:
    facts = payload.get("facts", {})
    evaluations = payload.get("evaluations", [])
    verdicts = payload.get("verdicts", [])
    lines: list[str] = []

    if not evaluations:
        lines.append("No scheme could be evaluated because no applicant details were provided.")
    for evaluation in evaluations:
        eligibility = evaluation.get("eligibility", {})
        status = eligibility.get("status", "unknown")
        failing = [r for r in eligibility.get("reasons", []) if r.get("status") == "fail"]
        missing = eligibility.get("missing_fields", [])
        sentence = f"Under {evaluation.get('name')}, the applicant is {_status_phrase(status)}."
        if failing:
            reasons = "; ".join(
                f"{r.get('description') or r.get('rule_id')} (actual {r.get('actual')} vs threshold {r.get('threshold')})"
                for r in failing[:3]
            )
            sentence += f" Failing criteria: {reasons}."
        if missing:
            sentence += f" Missing details: {', '.join(missing)}."
        lines.append(sentence)

    if verdicts:
        counts: dict[str, int] = {}
        for verdict in verdicts:
            counts[verdict.get("verification_status", "INSUFFICIENT_EVIDENCE")] = (
                counts.get(verdict.get("verification_status", "INSUFFICIENT_EVIDENCE"), 0) + 1
            )
        lines.append(
            "Evidence check: "
            + ", ".join(f"{count} {status.lower()}" for status, count in sorted(counts.items()))
            + "."
        )
    else:
        lines.append(
            "No indexed scheme document evidence was retrieved, so every claim is reported as "
            "INSUFFICIENT_EVIDENCE rather than guessed."
        )

    if facts.get("location_type") in {"rural", "urban"}:
        lines.append(
            f"The applicant is treated as {facts['location_type']}, which selects the "
            "corresponding income threshold where a scheme defines separate rural/urban limits."
        )
    lines.append(
        "All financial figures above come from the deterministic financial engine; this assistant "
        "never calculates EMI, interest or thresholds."
    )
    return " ".join(lines)


async def generate_explanation(
    payload: dict[str, Any],
    *,
    llm: LLMClient | None = None,
    use_live_llm: bool = True,
) -> ExplanationResult:
    allowed = collect_allowed_numbers(payload)
    template = template_explanation(payload)
    client = llm if llm is not None else (get_llm_client() if use_live_llm else None)

    if client is None:
        if use_live_llm:
            log_mock(logger, "explanation", "no GROQ_API_KEY - using template explanation")
        return ExplanationResult(text=template, provider="template", mock_mode=True)

    try:
        result = await client.complete(
            EXPLANATION_SYSTEM_PROMPT,
            "Deterministic facts (authoritative):\n"
            + _compact(payload)
            + "\n\nWrite the explanation now.",
        )
        ok, offending = numeric_guard(result.text, allowed)
        if not ok:
            log_event(
                logger,
                logging.WARNING,
                "LLM explanation failed the numeric guard - template used instead",
                offending=sorted(offending),
            )
            return ExplanationResult(
                text=template,
                provider="template",
                mock_mode=False,
                guard_triggered=True,
                warnings=[f"llm_numeric_guard_triggered: {sorted(offending)}"],
            )
        return ExplanationResult(text=result.text.strip(), provider=result.provider, mock_mode=False)
    except LLMError as exc:
        return ExplanationResult(
            text=template, provider="template", mock_mode=False, warnings=[f"llm_failed: {exc}"]
        )


def _compact(payload: dict[str, Any]) -> str:
    import json

    trimmed = {k: v for k, v in payload.items() if k != "trace"}
    return json.dumps(trimmed, default=str)[:12000]
