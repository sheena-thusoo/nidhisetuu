"""RAG package: chunking, embeddings, vector stores, retrieval, evidence verification."""

from app.rag.chunking import chunk_text, stable_chunk_id
from app.rag.retrieval import RagService, build_rag_service, get_rag_service, set_rag_service

__all__ = [
    "chunk_text",
    "stable_chunk_id",
    "RagService",
    "build_rag_service",
    "get_rag_service",
    "set_rag_service",
]
