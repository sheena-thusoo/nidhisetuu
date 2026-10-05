"""Integration-suite fixtures: fresh in-memory RAG service per test.

The RagService is a process-wide singleton; without a reset, chunks indexed by an
earlier test (possibly an older document version) leak into later assertions.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import HashEmbeddingProvider
from app.rag.retrieval import RagService, set_rag_service
from app.rag.vector_store import InMemoryVectorStore


@pytest.fixture(autouse=True)
def fresh_rag_service():
    service = RagService(store=InMemoryVectorStore(), embedder=HashEmbeddingProvider(dim=256), top_k=4)
    set_rag_service(service)
    yield service
    set_rag_service(None)
