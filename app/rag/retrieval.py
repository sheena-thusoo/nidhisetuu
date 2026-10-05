"""RAG service - indexing + retrieval with provenance and freshness on every hit."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.rag.chunking import stable_chunk_id
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.vector_store import VectorRecord, VectorStore, get_vector_store
from app.schemas.rag import ChunkMetadata, IndexedChunk, RawChunk, RetrievedChunk
from app.services.freshness import classify_freshness
from app.utils.logging import log_event

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider, *, top_k: int = 4):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    @property
    def mock_mode(self) -> bool:
        return bool(self.store.mock_mode or self.embedder.mock_mode)

    @property
    def provider(self) -> str:
        return f"{self.store.provider}+{self.embedder.provider}"

    async def index_chunks(
        self,
        *,
        scheme_id: str,
        scheme_version: int,
        content_hash: str,
        source_url: str | None,
        source_type: str,
        document_file: str | None,
        retrieved_at: str,
        raw_chunks: list[RawChunk],
        demo_data: bool = True,
    ) -> list[IndexedChunk]:
        if not raw_chunks:
            return []
        vectors = await self.embedder.embed_documents([chunk.text for chunk in raw_chunks])
        records: list[VectorRecord] = []
        indexed: list[IndexedChunk] = []
        for raw, vector in zip(raw_chunks, vectors):
            chunk_id = stable_chunk_id(scheme_id, scheme_version, content_hash, raw.chunk_index)
            metadata = ChunkMetadata(
                scheme_id=scheme_id,
                scheme_version=scheme_version,
                source_url=source_url,
                source_type=source_type,
                document_file=document_file,
                section=raw.section,
                chunk_index=raw.chunk_index,
                content_hash=content_hash,
                retrieved_at=retrieved_at,
                is_table_row=raw.is_table_row,
                demo_data=demo_data,
            )
            indexed.append(IndexedChunk(chunk_id=chunk_id, text=raw.text, metadata=metadata))
            records.append(
                VectorRecord(id=chunk_id, vector=vector, text=raw.text, metadata=metadata.model_dump())
            )
        await self.store.upsert(records)
        log_event(
            logger,
            logging.INFO,
            "indexed chunks",
            scheme_id=scheme_id,
            scheme_version=scheme_version,
            chunks=len(records),
            provider=self.provider,
            mock_mode=self.mock_mode,
        )
        return indexed

    async def delete_scheme(self, scheme_id: str, scheme_version: int | None = None) -> int:
        return await self.store.delete(scheme_id=scheme_id, scheme_version=scheme_version)

    async def count(self, scheme_id: str | None = None) -> int:
        return await self.store.count(scheme_id=scheme_id)

    async def search(
        self,
        query: str,
        *,
        scheme_id: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []
        vector = await self.embedder.embed_query(query)
        hits = await self.store.query(vector, top_k=top_k or self.top_k, scheme_id=scheme_id)
        results: list[RetrievedChunk] = []
        for hit in hits:
            if min_score is not None and hit.score < min_score:
                continue
            metadata = ChunkMetadata.model_validate(hit.metadata)
            results.append(
                RetrievedChunk(
                    chunk_id=hit.id,
                    text=hit.text,
                    score=round(hit.score, 6),
                    metadata=metadata,
                    freshness=classify_freshness(metadata.retrieved_at or ""),
                )
            )
        return results


_rag_service: RagService | None = None


def build_rag_service(settings=None) -> RagService:
    settings = settings or get_settings()
    return RagService(
        store=get_vector_store(settings),
        embedder=get_embedding_provider(settings),
        top_k=settings.rag_top_k,
    )


def set_rag_service(service: RagService | None) -> None:
    global _rag_service
    _rag_service = service


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = build_rag_service()
    return _rag_service
