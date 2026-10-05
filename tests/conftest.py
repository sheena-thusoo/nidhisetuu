"""Shared test fixtures.

Tests never touch a real MongoDB, Redis, Groq, Pinecone or Tavily endpoint:
  * Mongo  -> mongomock-motor in-memory database (same Motor async API);
  * Vector -> in-memory vector store;
  * LLM    -> deterministic mock provider (marked [MOCK MODE] in logs);
  * Jobs   -> inline executor.
Live-credential tests are marked and SKIPPED unless the matching key is configured.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENV", "test")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("PINECONE_API_KEY", "")
os.environ.setdefault("TAVILY_API_KEY", "")
os.environ.setdefault("REDIS_URL", "")

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Database, set_database  # noqa: E402
from app.rules.loader import get_rule_repository  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def rule_repo():
    return get_rule_repository()


@pytest.fixture
async def mongo_db():
    """Fresh in-memory Mongo database per test (motor-compatible)."""
    from mongomock_motor import AsyncMongoMockClient

    client = AsyncMongoMockClient()
    db = Database(client[f"nidhisetu_test_{os.getpid()}"])
    await db.ensure_indexes()
    set_database(db)
    yield db
    set_database(None)  # type: ignore[arg-type]


@pytest.fixture
def nfsdc_facts() -> dict:
    return {
        "annual_income": 80000,
        "category": "SC",
        "location_type": "rural",
        "age": 22,
        "course_level": "undergraduate",
        "state": "Bihar",
    }
