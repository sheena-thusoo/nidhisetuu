"""Evidence verification agent.

Given claims (from rule data, financial config and scheme fact statements) and retrieved
chunks, classify EACH claim as one of:
  SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT_EVIDENCE | CONFLICTING_EVIDENCE

Two verification paths:
  * deterministic lexical verification (always runs; the offline/mock-mode path);
  * optional Groq LLM classification, which is merged with - and can never override -
    the deterministic conflict detector. If the LLM is unavailable or returns junk,
    the lexical verdict is kept.

No supporting chunk => INSUFFICIENT_EVIDENCE. Citations are never fabricated.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from app.ai.json_utils import extract_json
from app.ai.llm import LLMClient, LLMError, get_llm_client
from app.schemas.rag import EvidenceClaim, EvidenceSource, EvidenceVerdict, RetrievedChunk
from app.services.eligibility import SchemeEvaluation
from app.utils.logging import log_event, log_mock

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "has", "have", "had",
    "not", "but", "from", "they", "their", "must", "should", "will", "shall", "may", "per",
    "annum", "under", "into", "also", "than", "then", "there", "been", "being", "its", "any",
    "all", "can", "could", "would", "about", "which", "who", "whom", "is", "of", "to", "in",
    "on", "at", "by", "or", "as", "be", "a", "an", "applicant", "applicants", "scheme",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_.-]+|\d+(?:\.\d+)?")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

SUPPORTED_OVERLAP = 0.25
PARTIAL_OVERLAP = 0.25
CONFLICT_OVERLAP = 0.5
SUPPORT_OVERLAP_NO_NUMBERS = 0.5
SNIPPET_LENGTH = 300
MAX_SOURCES = 3


def _normalize_number(token: str) -> str:
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


def extract_numbers(text: str) -> set[str]:
    return {_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text or "")}


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= length else cleaned[:length].rstrip() + "..."


# ------------------------------------------------------------------ claim building


def build_claims_for_scheme(evaluation: SchemeEvaluation) -> list[EvidenceClaim]:
    """Claims describe what the scheme data *asserts*, so they can be checked against sources."""
    claims: list[EvidenceClaim] = []
    scheme_id = evaluation.scheme_id

    for rule in evaluation.eligibility.reasons:
        description = rule.description or f"Rule {rule.rule_id} uses operator {rule.operator}."
        if rule.unit == "INR" and rule.threshold is not None and rule.threshold_variant:
            # threshold_map rules: verify the location-specific threshold that was applied
            text = f"{evaluation.name}: {description} Applied {rule.threshold_variant} threshold: {rule.threshold}."
        else:
            text = f"{evaluation.name}: {description}"
        if not text.strip():
            continue
        claims.append(
            EvidenceClaim(
                claim_id=f"{scheme_id}:rule:{rule.rule_id}",
                claim=text,
                claim_type="scheme_rule",
                scheme_id=scheme_id,
                rule_id=rule.rule_id,
                numbers=sorted(extract_numbers(text)),
            )
        )

    config = evaluation.financial_config
    if config is not None:
        financial_claims = [
            (f"{evaluation.name}: the interest rate is {config.interest_rate_annual_percent}% per annum.", "financials:interest_rate"),
            (f"{evaluation.name}: the maximum loan amount is INR {config.max_principal}.", "financials:max_amount"),
            (
                f"{evaluation.name}: repayment tenure options are {', '.join(str(m) for m in config.tenure_months_options)} months.",
                "financials:tenure_options",
            ),
        ]
        if config.processing_fee_percent:
            financial_claims.append(
                (f"{evaluation.name}: the processing fee is {config.processing_fee_percent}% of the loan amount.", "financials:processing_fee")
            )
        for text, suffix in financial_claims:
            claims.append(
                EvidenceClaim(
                    claim_id=f"{scheme_id}:{suffix}",
                    claim=text,
                    claim_type="financials",
                    scheme_id=scheme_id,
                    numbers=sorted(extract_numbers(text)),
                )
            )

    for index, text in enumerate(evaluation.claims):
        claims.append(
            EvidenceClaim(
                claim_id=f"{scheme_id}:fact:{index}",
                claim=text,
                claim_type="scheme_fact",
                scheme_id=scheme_id,
                numbers=sorted(extract_numbers(text)),
            )
        )
    return claims


# ------------------------------------------------------------- lexical verification


def _evidence_source(chunk: RetrievedChunk) -> EvidenceSource:
    return EvidenceSource(
        scheme_id=chunk.metadata.scheme_id,
        scheme_version=chunk.metadata.scheme_version,
        source_id=f"{chunk.metadata.scheme_id}:v{chunk.metadata.scheme_version}",
        source_url=chunk.metadata.source_url,
        document_file=chunk.metadata.document_file,
        section=chunk.metadata.section,
        page=chunk.metadata.page,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.metadata.chunk_index,
        retrieved_at=chunk.metadata.retrieved_at,
        freshness=chunk.freshness,
        snippet=snippet(chunk.text),
        score=chunk.score,
        is_table_row=chunk.metadata.is_table_row,
    )


def _classify_chunk(claim_tokens: set[str], claim_numbers: set[str], chunk_text: str) -> tuple[str, dict[str, Any]]:
    chunk_tokens = content_tokens(chunk_text)
    chunk_numbers = extract_numbers(chunk_text)
    overlap = (len(claim_tokens & chunk_tokens) / len(claim_tokens)) if claim_tokens else 0.0
    shared = claim_numbers & chunk_numbers
    missing = claim_numbers - chunk_numbers

    if claim_numbers:
        if not missing and overlap >= SUPPORTED_OVERLAP:
            return "supported", {"overlap": overlap}
        if shared and missing:
            return "partial", {"overlap": overlap}
        if not shared and overlap >= CONFLICT_OVERLAP:
            return "conflicting", {"overlap": overlap, "chunk_numbers": sorted(chunk_numbers)}
    else:
        if overlap >= SUPPORT_OVERLAP_NO_NUMBERS:
            return "supported", {"overlap": overlap}
        if overlap >= PARTIAL_OVERLAP:
            return "partial", {"overlap": overlap}
    return "none", {"overlap": overlap}


def lexical_verify(claim: EvidenceClaim, chunks: Iterable[RetrievedChunk]) -> EvidenceVerdict:
    claim_tokens = content_tokens(claim.claim)
    claim_numbers = set(claim.numbers) or extract_numbers(claim.claim)

    supported: list[tuple[float, EvidenceSource]] = []
    partial: list[tuple[float, EvidenceSource]] = []
    conflicting: list[tuple[float, EvidenceSource]] = []

    for chunk in chunks:
        kind, info = _classify_chunk(claim_tokens, claim_numbers, chunk.text)
        if kind == "none":
            continue
        source = _evidence_source(chunk)
        if kind == "supported":
            supported.append((info["overlap"], source))
        elif kind == "partial":
            partial.append((info["overlap"], source))
        else:
            conflicting.append((info["overlap"], source))

    supported.sort(key=lambda item: (-item[0], item[1].chunk_id))
    partial.sort(key=lambda item: (-item[0], item[1].chunk_id))
    conflicting.sort(key=lambda item: (-item[0], item[1].chunk_id))

    if conflicting:
        return EvidenceVerdict(
            claim_id=claim.claim_id,
            claim=claim.claim,
            claim_type=claim.claim_type,
            scheme_id=claim.scheme_id,
            verification_status="CONFLICTING_EVIDENCE",
            confidence=0.6,
            verified_by="lexical",
            matched_sources=[s for _, s in supported[:MAX_SOURCES]],
            conflicting_sources=[s for _, s in conflicting[:MAX_SOURCES]],
            notes="Retrieved sources disagree with the claim's numbers."
            + (" Other sources support it." if supported else ""),
            mock_mode=True,
        )
    if supported:
        best = supported[0][0]
        return EvidenceVerdict(
            claim_id=claim.claim_id,
            claim=claim.claim,
            claim_type=claim.claim_type,
            scheme_id=claim.scheme_id,
            verification_status="SUPPORTED",
            confidence=round(min(0.95, 0.6 + 0.35 * best), 2),
            verified_by="lexical",
            matched_sources=[s for _, s in supported[:MAX_SOURCES]],
            mock_mode=True,
        )
    if partial:
        return EvidenceVerdict(
            claim_id=claim.claim_id,
            claim=claim.claim,
            claim_type=claim.claim_type,
            scheme_id=claim.scheme_id,
            verification_status="PARTIALLY_SUPPORTED",
            confidence=0.45,
            verified_by="lexical",
            matched_sources=[s for _, s in partial[:MAX_SOURCES]],
            notes="Sources cover this claim only partially (some numbers or terms are missing).",
            mock_mode=True,
        )
    return EvidenceVerdict(
        claim_id=claim.claim_id,
        claim=claim.claim,
        claim_type=claim.claim_type,
        scheme_id=claim.scheme_id,
        verification_status="INSUFFICIENT_EVIDENCE",
        confidence=0.1,
        verified_by="lexical",
        matched_sources=[],
        notes="No retrieved chunk supports this claim - it is reported without a citation.",
        mock_mode=True,
    )


# --------------------------------------------------------------------------- LLM path

VERIFY_SYSTEM_PROMPT = (
    "You verify whether retrieved document chunks support a claim about an Indian government "
    "loan/education scheme. Reply ONLY with JSON: "
    '{"status": "SUPPORTED|PARTIALLY_SUPPORTED|INSUFFICIENT_EVIDENCE|CONFLICTING_EVIDENCE", '
    '"confidence": 0.0-1.0, "reason": "short"}.\n'
    "SUPPORTED requires the chunk to state the claim's numbers/terms explicitly. "
    "If numbers differ, use CONFLICTING_EVIDENCE. If the chunk does not address the claim, "
    "use INSUFFICIENT_EVIDENCE. Never guess."
)
_VALID_STATUSES = {"SUPPORTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}


async def _llm_verify_claim(claim: EvidenceClaim, chunks: list[RetrievedChunk], llm: LLMClient) -> EvidenceVerdict | None:
    context = "\n\n".join(
        f"[chunk {c.chunk_id} | section {c.metadata.section} | source {c.metadata.source_url}]\n{c.text[:700]}"
        for c in chunks[:4]
    )
    user = f"CLAIM: {claim.claim}\n\nRETRIEVED CHUNKS:\n{context or '(none retrieved)'}"
    result = await llm.complete(VERIFY_SYSTEM_PROMPT, user, json_mode=True)
    parsed = extract_json(result.text)
    if not isinstance(parsed, dict) or parsed.get("status") not in _VALID_STATUSES:
        return None
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return EvidenceVerdict(
        claim_id=claim.claim_id,
        claim=claim.claim,
        claim_type=claim.claim_type,
        scheme_id=claim.scheme_id,
        verification_status=parsed["status"],
        confidence=max(0.0, min(1.0, confidence)),
        verified_by="llm+lexical",
        notes=str(parsed.get("reason", ""))[:300] or None,
        mock_mode=False,
    )


async def verify_claims(
    claims: list[EvidenceClaim],
    chunks: list[RetrievedChunk],
    *,
    llm: LLMClient | None = None,
    use_live_llm: bool = True,
) -> list[EvidenceVerdict]:
    """Verify every claim. Deterministic lexical verdicts are always computed first."""
    verdicts = [lexical_verify(claim, chunks) for claim in claims]
    if not verdicts:
        return verdicts

    client = llm if llm is not None else (get_llm_client() if use_live_llm else None)
    if client is None:
        if use_live_llm:
            log_mock(logger, "evidence_verifier", "no GROQ_API_KEY - deterministic lexical verification only")
        return verdicts

    by_id = {v.claim_id: v for v in verdicts}
    for claim in claims:
        try:
            llm_verdict = await _llm_verify_claim(claim, chunks, client)
        except LLMError as exc:
            log_event(logger, logging.WARNING, "llm verification failed, lexical verdict kept", error=str(exc))
            continue
        if llm_verdict is None:
            continue
        lexical = by_id[claim.claim_id]
        merged = _merge_verdicts(lexical, llm_verdict)
        by_id[claim.claim_id] = merged
    return [by_id[verdict.claim_id] for verdict in verdicts]


def _merge_verdicts(lexical: EvidenceVerdict, llm_verdict: EvidenceVerdict) -> EvidenceVerdict:
    """Deterministic conflict detection always wins; otherwise prefer the LLM verdict."""
    if lexical.verification_status == "CONFLICTING_EVIDENCE":
        return lexical
    merged = llm_verdict.model_copy(deep=True)
    merged.matched_sources = lexical.matched_sources or llm_verdict.matched_sources
    merged.conflicting_sources = lexical.conflicting_sources
    merged.verified_by = "llm+lexical"
    if merged.verification_status == "SUPPORTED" and not merged.matched_sources:
        # the LLM claims support but the deterministic pass found no citation: never fabricate
        merged.verification_status = "INSUFFICIENT_EVIDENCE"
        merged.notes = "LLM indicated support but no deterministic citation was found."
        merged.confidence = min(merged.confidence, 0.3)
    return merged
