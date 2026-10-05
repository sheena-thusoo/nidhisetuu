"""Embedding providers.

MOCK MODE embeds deterministically with a hashing vectorizer: no network, no cost,
byte-identical output for identical input - good enough for offline tests and demos,
explicitly NOT a semantic model (it is lexical + numeric overlap).

LIVE MODE uses Pinecone's inference API (`multilingual-e5-large`, 1024 dims) which
handles Indian-language content well. Only used when PINECONE_API_KEY is set.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, Sequence

from app.utils.logging import log_mock

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    provider: str
    mock_mode: bool
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 or t.isdigit()]


class HashEmbeddingProvider:
    """Deterministic hashing vectorizer (mock mode)."""

    provider = "hash-embedding"
    mock_mode = True

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # numbers (amounts, rates, tenures) carry the most scheme-specific signal
            weight = 3.0 if token.isdigit() else 1.0 + math.log1p(len(token))
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class PineconeInferenceEmbeddingProvider:
    """Live embeddings via Pinecone inference (multilingual-e5-large)."""

    provider = "pinecone-inference"
    mock_mode = False

    def __init__(self, client, model: str = "multilingual-e5-large", dim: int = 1024):
        self._client = client
        self.model = model
        self.dim = dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.inference.embed(
            model=self.model, inputs=list(texts), parameters={"input_type": "passage", "truncate": "END"}
        )
        return [item["values"] for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        response = self._client.inference.embed(
            model=self.model, inputs=[text], parameters={"input_type": "query", "truncate": "END"}
        )
        return response.data[0]["values"]


def get_embedding_provider(settings) -> EmbeddingProvider:
    if settings.pinecone_enabled:
        from pinecone import Pinecone

        client = Pinecone(api_key=settings.pinecone_api_key)
        return PineconeInferenceEmbeddingProvider(
            client, model=settings.pinecone_embedding_model, dim=settings.pinecone_embedding_dim
        )
    log_mock(
        logger,
        "embeddings",
        f"PINECONE_API_KEY not set - using deterministic hash embeddings ({settings.mock_embedding_dim} dims). "
        "Lexical/numeric overlap only; not a semantic model.",
    )
    return HashEmbeddingProvider(dim=settings.mock_embedding_dim)
