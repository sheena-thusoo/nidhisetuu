"""Phase 7: the eval harness must pass its full golden set in mock mode."""

from __future__ import annotations

import pytest

from scripts.evaluate_demo import GOLDEN_SET, run_eval


@pytest.fixture(autouse=True)
def fresh_rag():
    from app.rag.embeddings import HashEmbeddingProvider
    from app.rag.retrieval import RagService, set_rag_service
    from app.rag.vector_store import InMemoryVectorStore

    set_rag_service(RagService(InMemoryVectorStore(), HashEmbeddingProvider(dim=256), top_k=4))
    yield
    set_rag_service(None)


async def test_golden_set_passes_completely():
    report = await run_eval()
    assert report["total"] == len(GOLDEN_SET) == 9
    assert report["all_passed"] is True, [r for r in report["results"] if not r["pass"]]
    assert report["passed"] == 9 and report["failed"] == 0
    assert report["mock_mode"]["groq"] is True  # deterministic mock mode


async def test_eval_cases_carry_their_expectations():
    for _, _, _, scheme_id, expected in GOLDEN_SET:
        assert expected in {"potentially_eligible", "not_eligible", "insufficient_data"}
        assert scheme_id.startswith("demo_")
