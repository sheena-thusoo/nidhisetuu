"""Evidence verifier tests (deterministic lexical path) + claim building from evaluations.

These run in mock mode by design: conftest forces GROQ_API_KEY="", so verify_claims
always returns the lexical verdict. The LLM merge path is exercised with fakes.
"""

from __future__ import annotations

from datetime import timedelta

from app.rag.chunking import stable_chunk_id
from app.rag.evidence import (
    build_claims_for_scheme,
    extract_numbers,
    lexical_verify,
    verify_claims,
)
from app.schemas.rag import ChunkMetadata, EvidenceClaim, RetrievedChunk
from app.services.eligibility import match_schemes
from app.services.freshness import classify_freshness
from app.utils.timeutil import utcnow


def _chunk(
    text: str,
    *,
    scheme_id: str = "demo_nfsdc_education_loan",
    index: int = 0,
    retrieved_days_ago: int = 10,
) -> RetrievedChunk:
    now = utcnow()
    content_hash = "ab" * 32
    metadata = ChunkMetadata(
        scheme_id=scheme_id,
        scheme_version=1,
        source_url="https://example.gov.in/demo",
        document_file="demo_nfsdc_education_loan.txt",
        section="Eligibility",
        chunk_index=index,
        content_hash=content_hash,
        retrieved_at=(now - timedelta(days=retrieved_days_ago)).isoformat(timespec="milliseconds"),
        is_table_row=False,
        demo_data=True,
    )
    return RetrievedChunk(
        chunk_id=stable_chunk_id(scheme_id, 1, content_hash, index),
        text=text,
        score=0.83,
        metadata=metadata,
        freshness=classify_freshness(metadata.retrieved_at),
    )


def _claim(text: str, numbers: list[str] | None = None, claim_id: str = "c1") -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim=text,
        claim_type="scheme_fact",
        scheme_id="demo_nfsdc_education_loan",
        numbers=numbers if numbers is not None else sorted(extract_numbers(text)),
    )


def test_matching_chunk_with_all_numbers_gives_supported():
    claim = _claim("The income limit for rural applicants is 81000 rupees per year.")
    chunk = _chunk(
        "Applicant Eligibility: The annual family income limit for rural applicants is INR 81,000 "
        "per year. Urban applicants may earn up to INR 1,03,000 per year."
    )
    verdict = lexical_verify(claim, [chunk])
    assert verdict.verification_status == "SUPPORTED"
    assert verdict.verified_by == "lexical"
    assert verdict.matched_sources and verdict.matched_sources[0].chunk_id == chunk.chunk_id
    assert verdict.confidence > 0.5
    assert verdict.mock_mode is True


def test_partial_number_match_gives_partially_supported():
    # claim has two numbers; chunk states only one of them
    claim = _claim("Income limit is 81000 rural and 103000 urban.", numbers=["81000", "103000"])
    chunk = _chunk("The rural income ceiling is 81000 rupees for these loans.")
    verdict = lexical_verify(claim, [chunk])
    assert verdict.verification_status == "PARTIALLY_SUPPORTED"


def test_no_chunks_gives_insufficient_evidence():
    claim = _claim("The interest rate is 10 percent per annum.", numbers=["10"])
    verdict = lexical_verify(claim, [])
    assert verdict.verification_status == "INSUFFICIENT_EVIDENCE"
    assert verdict.matched_sources == []
    assert "without a citation" in (verdict.notes or "")


def test_different_numbers_with_low_overlap_is_not_conflicting():
    # numbers differ BUT token overlap is far below CONFLICT (0.5) -> no citation at all
    claim = _claim("The yearly rate is 10 percent fixed.", numbers=["10"])
    chunk = _chunk("Fresh proposals go to the regional office for document verification.")
    assert lexical_verify(claim, [chunk]).verification_status == "INSUFFICIENT_EVIDENCE"


def test_shared_context_but_opposite_number_is_conflicting():
    # high lexical overlap (>= CONFLICT_OVERLAP 0.5) but the claim's number is absent
    # and different numbers are present -> deterministic conflict detector must fire
    claim = _claim(
        "The income limit for rural applicants is 81000 rupees per year and 103000 for urban."
    )
    chunk = _chunk(
        "Eligibility says the income limit for rural applicants is 99000 rupees per year "
        "and 123000 for urban as per the revised circular."
    )
    verdict = lexical_verify(claim, [chunk])
    assert verdict.verification_status == "CONFLICTING_EVIDENCE"
    assert verdict.conflicting_sources, "conflict must cite the contradicting chunk"


def test_numberless_claim_supported_only_above_half_overlap():
    claim = _claim("Loans are available for diploma courses.", numbers=[])
    no_overlap = _chunk("The monsoon seasonal schedule is published separately this month.")
    some_overlap = _chunk("Loans are also provided for diploma courses at ITIs this cycle.")
    assert lexical_verify(claim, [no_overlap]).verification_status == "INSUFFICIENT_EVIDENCE"
    assert lexical_verify(claim, [some_overlap]).verification_status in {"SUPPORTED", "PARTIALLY_SUPPORTED"}


def test_verdicts_are_sorted_and_citations_carry_provenance():
    claim = _claim("The income limit for rural applicants is 81000 rupees per year.")
    newer = _chunk("The annual income limit for rural applicants is INR 81,000 per year.", index=2)
    verdict = lexical_verify(claim, [newer])
    source = verdict.matched_sources[0]
    assert source.scheme_id == "demo_nfsdc_education_loan"
    assert source.scheme_version == 1
    assert source.section == "Eligibility"
    assert source.source_url == "https://example.gov.in/demo"
    assert source.freshness.classification == "FRESH"
    assert source.retrieved_at.endswith("Z") is False  # plain ISO from utcnow()


def test_extract_numbers_normalization():
    assert extract_numbers("loan of 1,50,000 at 10.50% for 60 months") == {"150000", "10.5", "60"}


def test_build_claims_for_scheme_covers_rules_financials_and_facts(rule_repo):
    facts = {
        "annual_income": 80000,
        "category": "SC",
        "location_type": "rural",
        "age": 22,
        "course_level": "undergraduate",
    }
    evaluation = match_schemes(rule_repo, facts, scheme_ids=["demo_nfsdc_education_loan"])[0]
    claims = build_claims_for_scheme(evaluation)

    claim_types = {c.claim_type for c in claims}
    assert claim_types == {"scheme_rule", "financials", "scheme_fact"}
    rule_ids = {c.rule_id for c in claims if c.claim_type == "scheme_rule"}
    assert {"category", "income_limit", "age_limit", "course_level"} <= rule_ids
    # financial config claims carry the scheme's actual configured numbers
    financial_claims = [c for c in claims if c.claim_type == "financials"]
    assert any("10" in c.numbers for c in financial_claims)  # 10% interest rate
    assert any("1500000" in c.numbers for c in financial_claims)  # max amount
    # every claim id is namespaced by scheme id
    assert all(c.claim_id.startswith("demo_nfsdc_education_loan:") for c in claims)


def test_threshold_map_claim_records_applied_variant(rule_repo):
    facts = {
        "annual_income": 80000,
        "category": "SC",
        "location_type": "rural",
        "age": 22,
        "course_level": "undergraduate",
    }
    evaluation = match_schemes(rule_repo, facts, scheme_ids=["demo_nfsdc_education_loan"])[0]
    claims = build_claims_for_scheme(evaluation)
    income_claims = [c for c in claims if c.rule_id == "income_limit"]
    assert income_claims, "threshold_map rule must produce a claim"
    assert any("rural" in c.claim.lower() and "81000" in c.numbers for c in income_claims)


async def test_verify_claims_mock_mode_returns_lexical_verdicts():
    claims = [
        _claim("The interest rate is 10 percent per annum.", ["10"], claim_id="r1"),
        _claim("Repayment tenure options include 60 months.", ["60"], claim_id="r2"),
    ]
    chunks = [
        _chunk("The education loan interest rate is 10% per annum with tenure up to 60 months."),
        _chunk("Unrelated paragraph about application forms and offices."),
    ]
    verdicts = await verify_claims(claims, chunks)  # use_live_llm=True but no key -> lexical
    statuses = {v.claim_id: v.verification_status for v in verdicts}
    assert statuses["r1"] == "SUPPORTED"
    assert statuses["r2"] in {"SUPPORTED", "PARTIALLY_SUPPORTED"}


async def test_verify_claims_empty_input_returns_empty():
    assert await verify_claims([], []) == []


# ------------------------------------------------------------------ LLM merge path


class _FakeLLM:
    provider = "fake"
    model = "fake-model"
    mock_mode = False

    def __init__(self, replies: dict[str, str]):
        # keyed by a substring of the claim text, which is what appears in the prompt
        self._replies = replies

    async def complete(self, system, user, *, json_mode=False, temperature=0.0):
        from app.ai.llm import LLMResult

        for key, reply in self._replies.items():
            if key in user:
                return LLMResult(text=reply, provider=self.provider, model=self.model)
        return LLMResult(text="{}", provider=self.provider, model=self.model)


async def test_llm_supported_without_citation_is_downgraded():
    claim_text = "The rate is 10 percent."
    claims = [_claim(claim_text, ["10"], claim_id="r1")]
    chunks = [_chunk("This section talks about application timelines only.")]
    llm = _FakeLLM({claim_text: '{"status": "SUPPORTED", "confidence": 0.9, "reason": "looks right"}'})
    verdicts = await verify_claims(claims, chunks, llm=llm)
    assert verdicts[0].verification_status == "INSUFFICIENT_EVIDENCE"
    assert verdicts[0].notes == "LLM indicated support but no deterministic citation was found."
    assert verdicts[0].verified_by == "llm+lexical"
    assert verdicts[0].confidence <= 0.3


async def test_lexical_conflict_always_wins_over_llm():
    claim_text = (
        "The income limit for rural applicants is 81000 rupees per year and 103000 for urban."
    )
    claim = _claim(claim_text)
    chunk = _chunk(
        "Eligibility says the income limit for rural applicants is 99000 rupees per year "
        "and 123000 for urban as per the revised circular."
    )
    llm = _FakeLLM({claim_text: '{"status": "SUPPORTED", "confidence": 0.99, "reason": "sure"}'})
    verdicts = await verify_claims([claim], [chunk], llm=llm)
    assert verdicts[0].verification_status == "CONFLICTING_EVIDENCE"


async def test_llm_invalid_payload_falls_back_to_lexical():
    claim_text = "The rate is 10 percent."
    claims = [_claim(claim_text, ["10"], claim_id="r1")]
    chunks = [_chunk("The interest rate is 10% per annum.")]
    llm = _FakeLLM({claim_text: "this is not json at all"})
    verdicts = await verify_claims(claims, chunks, llm=llm)
    assert verdicts[0].verification_status == "SUPPORTED"
    assert verdicts[0].verified_by == "lexical"
