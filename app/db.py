"""MongoDB access layer (Motor, async).

The app talks to Mongo through a single `Database` wrapper. Tests inject an
in-memory mongomock-motor database through `set_database()`, so the exact same
queries run against real Mongo in production and a mock in CI.
"""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from app.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAMES = (
    "users",
    "schemes",
    "scheme_versions",
    "documents",
    "ingestion_jobs",
    "applications",
    "audit_events",
)


class Database:
    """Thin wrapper so callers never reach for a global client directly."""

    def __init__(self, db: Any):
        self._db = db

    @property
    def raw(self) -> Any:
        return self._db

    def __getitem__(self, name: str) -> Any:
        return self._db[name]

    @property
    def users(self) -> Any:
        return self._db["users"]

    @property
    def schemes(self) -> Any:
        return self._db["schemes"]

    @property
    def scheme_versions(self) -> Any:
        return self._db["scheme_versions"]

    @property
    def documents(self) -> Any:
        return self._db["documents"]

    @property
    def ingestion_jobs(self) -> Any:
        return self._db["ingestion_jobs"]

    @property
    def applications(self) -> Any:
        return self._db["applications"]

    @property
    def audit_events(self) -> Any:
        return self._db["audit_events"]

    async def ensure_indexes(self) -> None:
        await self.users.create_index("email", unique=True)
        await self.schemes.create_index("scheme_id", unique=True)
        await self.scheme_versions.create_index([("scheme_id", 1), ("version", 1)], unique=True)
        await self.scheme_versions.create_index([("scheme_id", 1), ("content_hash", 1)])
        await self.documents.create_index([("scheme_id", 1), ("version", 1)], unique=True)
        await self.ingestion_jobs.create_index("status")
        await self.ingestion_jobs.create_index("scheme_id")
        await self.applications.create_index([("user_id", 1), ("created_at", -1)])
        await self.applications.create_index([("user_id", 1), ("scheme_id", 1)])
        await self.audit_events.create_index([("created_at", -1)])


_current: Database | None = None
_client: AsyncIOMotorClient | None = None


def set_database(database: Database) -> None:
    """Used by the app lifespan (real Mongo) and by tests (mongomock-motor)."""
    global _current
    _current = database


def get_database() -> Database:
    if _current is None:
        raise RuntimeError("Database not initialised. Call init_database() during app startup.")
    return _current


def init_database(uri: str | None = None, db_name: str | None = None) -> Database:
    global _client
    settings = get_settings()
    uri = uri or settings.mongodb_uri
    db_name = db_name or settings.mongodb_db
    _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000, uuidRepresentation="standard")
    database = Database(_client[db_name])
    set_database(database)
    logger.info("mongo client initialised", extra={"extra_fields": {"db": db_name}})
    return database


async def close_database() -> None:
    global _client, _current
    if _client is not None:
        _client.close()
        _client = None
    _current = None
