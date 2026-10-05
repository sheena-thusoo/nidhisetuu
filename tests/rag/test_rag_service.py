"""RagService tests: indexing, retrieval ranking, scheme filtering, freshness propagation,
mock-mode flags, and delete/count semantics - all over the in-memory store (mock mode)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.rag.chunking import stable_chunk_id
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.retrieval import RagService, get_rag_service, set_rag_service
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.rag import RawChunk
from app.utils.timeutil import utcnow

SCHEME = "demo_nfsdc_education_loan"


def _service(dim: int = 256) -> RagService:
    return RagService(store=InMemoryVectorStore(), embedder=HashEmbeddingProvider(dim=dim), top_k=4)


def _raw(index: int, text: str) -> RawChunk:
    return RawChunk(chunk_index=index, text=text, section=None, is_table_row=False)


async def test_index_and_search_roundtrip():
    service = _service()
    indexed = await service.index_chunks(
        scheme_id=SCHEME,
        scheme_version=1,
        content_hash="ab" * 32,
        source_url="https://example.gov.in/x",
        source_type="file",
        document_file="x.txt",
        retrieved_at=utcnow().isoformat(),
        raw_chunks=[
            _raw(0, "The annual family income limit for rural applicants is INR 81,000 per year."),
            _raw(1, "The education loan interest rate is 10% per annum for all approved courses."),
        ],
        demo_data=True,
    )
    assert len(indexed) == 2
    assert all(c.chunk_id == stable_chunk_id(SCHEME, 1, "ab" * 32, i) for i, c in enumerate(indexed))

    hits = await service.search("rural income limit 81000", scheme_id=SCHEME, top_k=2)
    assert hits, "identical lexical content must retrieve in mock mode"
    assert all(h.metadata.scheme_id == SCHEME for h in hits)
    assert all(h.metadata.content_hash == "ab" * 32 for h in hits)
    # scores sorted descending
    assert hits[0].score >= hits[-1].score
    # freshness is attached to every hit
    assert hits[0].freshness.classification == "FRESH"
    assert hits[0].freshness.age_days < 1


async def test_search_empty_query_returns_empty():
    service = _service()
    assert await service.search("   ") == []


async def test_scheme_filter_excludes_other_schemes():
    service = _service()
    common_time = utcnow().isoformat()
    await service.index_chunks(
        scheme_id="scheme_a", scheme_version=1, content_hash="a" * 64,
        source_url=None, source_type="file", document_file="a.txt",
        retrieved_at=common_time,
        raw_chunks=[_raw(0, "Income limit is 81000 rupees for rural families.")],
    )
    await service.index_chunks(
        scheme_id="scheme_b", scheme_version=1, content_hash="b" * 64,
        source_url=None, source_type="file", document_file="b.txt",
        retrieved_at=common_time,
        raw_chunks=[_raw(0, "Income limit is 81000 rupees for rural families.")],
    )
    assert await service.count() == 2
    hits_a = await service.search("income limit 81000", scheme_id="scheme_a")
    assert {h.metadata.scheme_id for h in hits_a} == {"scheme_a"}
    unfiltered = await service.search("income limit 81000")
    assert {h.metadata.scheme_id for h in unfiltered} == {"scheme_a", "scheme_b"}


async def test_reingest_same_version_overwrites_not_duplicates():
    service = _service()
    kwargs = dict(
        scheme_id=SCHEME, scheme_version=1, content_hash="ab" * 32,
        source_url="u", source_type="file", document_file="x.txt",
        retrieved_at=utcnow().isoformat(),
    )
    await service.index_chunks(raw_chunks=[_raw(0, "Rate is 10 percent per annum.")], **kwargs)
    await service.index_chunks(raw_chunks=[_raw(0, "Rate is 10 percent per annum (updated).")], **kwargs)
    assert await service.count(scheme_id=SCHEME) == 1, "stable chunk ids must make re-ingestion idempotent"


async def test_aging_document_is_flagged_aging():
    service = _service()
    old = (utcnow() - timedelta(days=250)).isoformat()
    await service.index_chunks(
        scheme_id=SCHEME, scheme_version=1, content_hash="cd" * 32,
        source_url=None, source_type="file", document_file="x.txt",
        retrieved_at=old,
        raw_chunks=[_raw(0, "Interest rate is 10 percent per annum.")],
    )
    hits = await service.search("interest rate 10 percent")
    assert hits[0].freshness.classification == "AGING"
    assert 240 < hits[0].freshness.age_days < 260


async def test_mock_mode_flags_and_provider_labels():
    service = _service()
    assert service.mock_mode is True
    assert service.provider == "memory+hash-embedding"


async def test_delete_scheme_removes_only_that_scheme():
    service = _service()
    now = utcnow().isoformat()
    await service.index_chunks(
        scheme_id="del_a", scheme_version=1, content_hash="a" * 64, source_url=None,
        source_type="file", document_file="a.txt", retrieved_at=now,
        raw_chunks=[_raw(0, "Text for scheme a")],
    )
    await service.index_chunks(
        scheme_id="del_b", scheme_version=3, content_hash="b" * 64, source_url=None,
        source_type="file", document_file="b.txt", retrieved_at=now,
        raw_chunks=[_raw(0, "Text for scheme b")],
    )
    assert await service.delete_scheme("del_a") == 1
    assert await service.count() == 1
    assert await service.count(scheme_id="del_b") == 1


def test_hash_embedding_is_deterministic_and_numeric_weighted():
    embedder = HashEmbeddingProvider(dim=256)
    import asyncio

    v1 = asyncio.run(embedder.embed_query("income limit 81000"))
    v2 = asyncio.run(embedder.embed_query("income limit 81000"))
    assert v1 == v2
    assert len(v1) == 256
    # unit-normalised (or zero vector for empty input)
    norm = sum(x * x for x in v1) ** 0.5
    assert norm == pytest.approx(1.0)
    assert asyncio.run(embedder.embed_query("")) == [0.0] * 256


def test_singleton_setter_roundtrip():
    service = _service()
    set_rag_service(service)
    assert get_rag_service() is service
    set_rag_service(None)


async def test_top_k_limit_respected():
    service = RagService(store=InMemoryVectorStore(), embedder=HashEmbeddingProvider(dim=256), top_k=2)
    now = utcnow().isoformat()
    chunks = [_raw(i, f"Document sentence number {i} mentions rupees 1000{i} for fees.") for i in range(5)]
    await service.index_chunks(
        scheme_id=SCHEME, scheme_version=1, content_hash="ef" * 32, source_url=None,
        source_type="file", document_file="x.txt", retrieved_at=now, raw_chunks=chunks,
    )
    hits = await service.search("document sentence rupees fees", top_k=2)
    assert len(hits) == 2
