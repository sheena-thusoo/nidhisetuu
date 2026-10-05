"""RAG / evidence schemas: chunks, retrieved sources, claims and verification verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.services.freshness import Freshness

VerificationStatus = Literal[
    "SUPPORTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"
]


class ChunkMetadata(BaseModel):
    scheme_id: str
    scheme_version: int
    source_url: str | None = None
    source_type: str = "file"  # file | url
    document_file: str | None = None
    section: str | None = None
    page: int | None = None
    chunk_index: int = 0
    content_hash: str = ""
    retrieved_at: str = ""  # when the source document was fetched/ingested
    is_table_row: bool = False
    demo_data: bool = True


class RawChunk(BaseModel):
    chunk_index: int
    text: str
    section: str | None = None
    is_table_row: bool = False
    char_start: int = 0
    char_end: int = 0


class IndexedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata
    freshness: Freshness


class EvidenceSource(BaseModel):
    """Provenance for one piece of evidence: scheme -> version -> source -> section -> retrieved_at."""

    scheme_id: str
    scheme_version: int
    source_id: str
    source_url: str | None = None
    document_file: str | None = None
    section: str | None = None
    page: int | None = None
    chunk_id: str
    chunk_index: int
    retrieved_at: str
    freshness: Freshness
    snippet: str
    score: float
    is_table_row: bool = False


class EvidenceClaim(BaseModel):
    claim_id: str
    claim: str
    claim_type: Literal["scheme_rule", "financials", "scheme_fact", "external"] = "scheme_fact"
    scheme_id: str | None = None
    rule_id: str | None = None
    numbers: list[str] = Field(default_factory=list)


class EvidenceVerdict(BaseModel):
    claim_id: str
    claim: str
    claim_type: str
    scheme_id: str | None = None
    verification_status: VerificationStatus
    confidence: float = 0.0
    verified_by: str = "lexical"  # lexical | llm | hybrid
    matched_sources: list[EvidenceSource] = Field(default_factory=list)
    conflicting_sources: list[EvidenceSource] = Field(default_factory=list)
    notes: str | None = None
    mock_mode: bool = False


class RetrievalBundle(BaseModel):
    scheme_id: str | None = None
    query: str
    provider: str = "memory"
    mock_mode: bool = False
    chunks: list[RetrievedChunk] = Field(default_factory=list)
