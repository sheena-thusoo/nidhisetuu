"""Vector stores: in-memory (tests / mock mode) and Pinecone (live)."""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

from app.utils.logging import log_mock

logger = logging.getLogger(__name__)


class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any]


class ScoredVector(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    provider: str
    mock_mode: bool

    async def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    async def query(
        self, vector: list[float], *, top_k: int = 4, scheme_id: str | None = None
    ) -> list[ScoredVector]: ...

    async def delete(self, *, scheme_id: str, scheme_version: int | None = None) -> int: ...

    async def count(self, *, scheme_id: str | None = None) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Deterministic cosine-similarity store used by tests and mock mode."""

    provider = "memory"
    mock_mode = True

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            self._records[record.id] = record
        return len(records)

    async def query(
        self, vector: list[float], *, top_k: int = 4, scheme_id: str | None = None
    ) -> list[ScoredVector]:
        scored: list[ScoredVector] = []
        for record in self._records.values():
            if scheme_id and record.metadata.get("scheme_id") != scheme_id:
                continue
            score = _cosine(vector, record.vector)
            scored.append(ScoredVector(id=record.id, score=round(score, 6), text=record.text, metadata=record.metadata))
        scored.sort(key=lambda item: (-item.score, item.id))
        return scored[:top_k]

    async def delete(self, *, scheme_id: str, scheme_version: int | None = None) -> int:
        doomed = [
            rid
            for rid, record in self._records.items()
            if record.metadata.get("scheme_id") == scheme_id
            and (scheme_version is None or record.metadata.get("scheme_version") == scheme_version)
        ]
        for rid in doomed:
            del self._records[rid]
        return len(doomed)

    async def count(self, *, scheme_id: str | None = None) -> int:
        if scheme_id is None:
            return len(self._records)
        return sum(1 for r in self._records.values() if r.metadata.get("scheme_id") == scheme_id)

    def reset(self) -> None:
        self._records.clear()


class PineconeVectorStore:
    """Live Pinecone store. Metadata mirrors the chunk metadata 1:1 for provenance filtering."""

    provider = "pinecone"
    mock_mode = False

    def __init__(self, api_key: str, index_name: str, dim: int):
        from pinecone import Pinecone

        self._client = Pinecone(api_key=api_key)
        self._index = self._client.Index(index_name)
        self._dim = dim

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        payload = [
            {"id": r.id, "values": r.vector, "metadata": {**r.metadata, "text": r.text}} for r in records
        ]
        for start in range(0, len(payload), 100):
            self._index.upsert(vectors=payload[start : start + 100])
        return len(payload)

    async def query(
        self, vector: list[float], *, top_k: int = 4, scheme_id: str | None = None
    ) -> list[ScoredVector]:
        filter_expr = {"scheme_id": {"$eq": scheme_id}} if scheme_id else None
        response = self._index.query(
            vector=vector, top_k=top_k, include_metadata=True, filter=filter_expr
        )
        results: list[ScoredVector] = []
        for match in response.get("matches", []):
            metadata = dict(match.get("metadata") or {})
            text = metadata.pop("text", "")
            results.append(ScoredVector(id=str(match["id"]), score=float(match["score"]), text=text, metadata=metadata))
        return results

    async def delete(self, *, scheme_id: str, scheme_version: int | None = None) -> int:
        filter_expr: dict[str, Any] = {"scheme_id": {"$eq": scheme_id}}
        if scheme_version is not None:
            filter_expr = {
                "$and": [
                    {"scheme_id": {"$eq": scheme_id}},
                    {"scheme_version": {"$eq": scheme_version}},
                ]
            }
        # Pinecone delete-by-filter is synchronous in the SDK; count is not returned.
        self._index.delete(filter=filter_expr)
        return 0

    async def count(self, *, scheme_id: str | None = None) -> int:
        try:
            stats = self._index.describe_index_stats()
            if scheme_id is None:
                return int(stats.get("total_vector_count", 0))
            namespaces = stats.get("namespaces", {}) or {}
            return int(sum(ns.get("vector_count", 0) for ns in namespaces.values()))
        except Exception:  # noqa: BLE001 - stats are best-effort
            return 0


def get_vector_store(settings) -> VectorStore:
    if settings.pinecone_enabled:
        return PineconeVectorStore(settings.pinecone_api_key or "", settings.pinecone_index, settings.pinecone_embedding_dim)
    log_mock(
        logger,
        "vector_store",
        f"PINECONE_API_KEY not set - using in-memory vector store for index '{settings.pinecone_index}'.",
    )
    return InMemoryVectorStore()
