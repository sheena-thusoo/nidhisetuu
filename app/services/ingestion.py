"""Ingestion pipeline: parse -> hash -> version -> chunk -> embed -> upsert.

Job execution model:
  * Redis/RQ available -> the job is enqueued (queue name from settings);
  * otherwise (or when settings.force_inline_jobs) -> the SAME pipeline function runs
    inline in-process, and the job document is updated in place.

Every job document lives in the `ingestion_jobs` Mongo collection with a full
status trail so an admin can see exactly what happened (including mock-mode flags
at the time of execution).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.db import Database, get_database
from app.rag.parsing import ParsedDocument, parse_document
from app.rag.retrieval import get_rag_service, set_rag_service
from app.rag.chunking import chunk_text
from app.rag.embeddings import get_embedding_provider
from app.rag.vector_store import get_vector_store
from app.utils.ids import new_job_id
from app.utils.logging import log_event
from app.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

JOB_STATUSES = ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "SKIPPED")


def compute_content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass
class IngestOutcome:
    job_id: str
    status: str
    scheme_id: str
    version: int | None
    content_hash: str
    chunks_indexed: int
    version_bumped: bool
    executed_inline: bool
    error: str | None = None


# --------------------------------------------------------------------------- versioning


async def _resolve_version(
    db: Database,
    *,
    scheme_id: str,
    content_hash: str,
    source_url: str | None,
    source_type: str,
    document_file: str | None,
    demo_data: bool,
) -> tuple[int, bool]:
    """Return (version, bumped). Same content -> same version; changed content -> +1.

    Unversioned documents (ingested before this field existed) are treated as
    version 0 and upgraded to 1 on the next ingestion.
    """
    versions = db.scheme_versions
    existing_same = await versions.find_one({"scheme_id": scheme_id, "content_hash": content_hash})
    if existing_same:
        return int(existing_same.get("version", 1)), False

    latest = await versions.find_one({"scheme_id": scheme_id}, sort=[("version", -1)])
    prev_version = int(latest.get("version", 0)) if latest else 0
    new_version = prev_version + 1
    await versions.insert_one(
        {
            "scheme_id": scheme_id,
            "version": new_version,
            "content_hash": content_hash,
            "source_url": source_url,
            "source_type": source_type,
            "document_file": document_file,
            "demo_data": demo_data,
            "first_seen_at": utcnow().isoformat(),
        }
    )
    await db.schemes.update_one(
        {"scheme_id": scheme_id},
        {
            "$set": {
                "scheme_id": scheme_id,
                "latest_version": new_version,
                "content_hash": content_hash,
                "source_url": source_url,
                "updated_at": utcnow().isoformat(),
            }
        },
        upsert=True,
    )
    return new_version, True


# --------------------------------------------------------------------------- the pipeline


async def run_ingestion(
    *,
    job_id: str,
    scheme_id: str,
    filename: str,
    raw_bytes: bytes,
    source_url: str | None = None,
    source_type: str = "file",
    demo_data: bool = True,
    db: Database | None = None,
    settings: Settings | None = None,
) -> IngestOutcome:
    """The ingestion pipeline proper. Runs inline OR inside an RQ worker.

    Raises only on unexpected errors; expected problems (bad scheme_id, unsupported
    type, no parsable content) mark the job FAILED/SKIPPED and return normally.
    """
    db = db or get_database()
    settings = settings or get_settings()
    jobs = db.ingestion_jobs

    await jobs.update_one({"job_id": job_id}, {"$set": {"status": "RUNNING", "started_at": utcnow().isoformat()}})
    try:
        from app.rules.loader import get_rule_repository

        if scheme_id not in get_rule_repository().ids():
            # rule JSON files in data/scheme_rules are the canonical scheme registry
            await jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "FAILED",
                    "finished_at": utcnow().isoformat(),
                    "error": f"unknown scheme_id {scheme_id!r} (not present in data/scheme_rules)",
                }},
            )
            return IngestOutcome(job_id, "FAILED", scheme_id, None, "", 0, False, True, "unknown scheme_id")

        # 1. parse --------------------------------------------------------------
        import tempfile
        from pathlib import Path

        from app.rag.parsing import UnsupportedDocumentType

        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(raw_bytes)
            temp_path = Path(handle.name)
        try:
            try:
                parsed: ParsedDocument = parse_document(temp_path)
            except UnsupportedDocumentType as exc:
                await jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {"status": "FAILED", "finished_at": utcnow().isoformat(), "error": str(exc)}},
                )
                return IngestOutcome(job_id, "FAILED", scheme_id, None, "", 0, False, True, str(exc))
        finally:
            temp_path.unlink(missing_ok=True)

        # meta fallbacks from the document itself (HTML <meta name="scheme_id"> etc.)
        parsed_scheme_id = parsed.scheme_id or scheme_id
        if parsed_scheme_id != scheme_id:
            await jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "FAILED",
                    "finished_at": utcnow().isoformat(),
                    "error": f"document meta scheme_id {parsed_scheme_id!r} != requested {scheme_id!r}",
                }},
            )
            return IngestOutcome(job_id, "FAILED", scheme_id, None, "", 0, False, True, "scheme_id mismatch")
        source_url = parsed.source_url or source_url

        # 2. hash + version ------------------------------------------------------
        content_hash = compute_content_hash(raw_bytes)
        version, bumped = await _resolve_version(
            db,
            scheme_id=scheme_id,
            content_hash=content_hash,
            source_url=source_url,
            source_type=source_type,
            document_file=filename,
            demo_data=demo_data,
        )

        # 3. chunk ----------------------------------------------------------------
        raw_chunks = chunk_text(parsed.text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not raw_chunks:
            await jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "SKIPPED",
                    "finished_at": utcnow().isoformat(),
                    "error": "document produced no chunks (empty content?)",
                }},
            )
            return IngestOutcome(job_id, "SKIPPED", scheme_id, version, content_hash, 0, bumped, True, "no chunks")

        # 4. embed + upsert --------------------------------------------------------
        rag = get_rag_service()
        indexed = await rag.index_chunks(
            scheme_id=scheme_id,
            scheme_version=version,
            content_hash=content_hash,
            source_url=source_url,
            source_type=source_type,
            document_file=filename,
            retrieved_at=utcnow().isoformat(),
            raw_chunks=raw_chunks,
            demo_data=demo_data,
        )
        if bumped and version > 1:
            # supersede the previous version's chunks so retrieval reflects the
            # current document only; prior versions stay auditable in Mongo
            await rag.delete_scheme(scheme_id, scheme_version=version - 1)

        # 5. record the document -----------------------------------------------------
        await db.documents.update_one(
            {"scheme_id": scheme_id, "version": version},
            {
                "$set": {
                    "scheme_id": scheme_id,
                    "version": version,
                    "content_hash": content_hash,
                    "document_file": filename,
                    "source_url": source_url,
                    "source_type": source_type,
                    "chunks": len(indexed),
                    "parser": parsed.parser,
                    "bytes": len(raw_bytes),
                    "ingested_at": utcnow().isoformat(),
                    "demo_data": demo_data,
                }
            },
            upsert=True,
        )
        await jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "COMPLETED",
                "finished_at": utcnow().isoformat(),
                "version": version,
                "content_hash": content_hash,
                "chunks_indexed": len(indexed),
                "version_bumped": bumped,
                "executed_mode": "inline" if not settings.redis_enabled else "rq_queue",
                "mock_mode": {
                    "redis_rq": not settings.redis_enabled,
                    "vector_store": rag.mock_mode,
                    "llm": not settings.groq_enabled,
                },
            }}
        )
        log_event(
            logger, logging.INFO, "ingestion completed",
            job_id=job_id, scheme_id=scheme_id, version=version,
            chunks=len(indexed), version_bumped=bumped, mock_mode=rag.mock_mode,
        )
        return IngestOutcome(job_id, "COMPLETED", scheme_id, version, content_hash, len(indexed), bumped, True)

    except Exception as exc:  # noqa: BLE001 - any pipeline error fails the job, never the process
        logger.exception("ingestion job crashed")
        await jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "FAILED", "finished_at": utcnow().isoformat(), "error": str(exc)[:500]}},
        )
        return IngestOutcome(job_id, "FAILED", scheme_id, None, "", 0, False, True, str(exc)[:500])


def execute_ingestion_job(
    job_id: str,
    scheme_id: str,
    filename: str,
    raw_bytes: bytes,
    source_url: str | None,
    source_type: str,
    demo_data: bool,
) -> dict[str, Any]:
    """RQ entrypoint (sync). Bridges the sync worker into the async pipeline."""
    import asyncio

    outcome = asyncio.run(
        run_ingestion(
            job_id=job_id,
            scheme_id=scheme_id,
            filename=filename,
            raw_bytes=raw_bytes,
            source_url=source_url,
            source_type=source_type,
            demo_data=demo_data,
        )
    )
    return {
        "job_id": outcome.job_id,
        "status": outcome.status,
        "scheme_id": outcome.scheme_id,
        "version": outcome.version,
        "content_hash": outcome.content_hash,
        "chunks_indexed": outcome.chunks_indexed,
        "version_bumped": outcome.version_bumped,
        "error": outcome.error,
    }


# --------------------------------------------------------------------------- job creation


async def create_and_dispatch(
    *,
    scheme_id: str,
    filename: str,
    raw_bytes: bytes,
    source_url: str | None = None,
    source_type: str = "file",
    demo_data: bool = True,
    db: Database | None = None,
    settings: Settings | None = None,
) -> IngestOutcome:
    """Create the job document, then enqueue OR run inline. Returns the outcome."""
    db = db or get_database()
    settings = settings or get_settings()
    job_id = new_job_id()

    await db.ingestion_jobs.insert_one(
        {
            "job_id": job_id,
            "scheme_id": scheme_id,
            "document_file": filename,
            "status": "QUEUED",
            "content_hash": compute_content_hash(raw_bytes),
            "source_url": source_url,
            "source_type": source_type,
            "bytes": len(raw_bytes),
            "created_at": utcnow().isoformat(),
            "mock_mode": {"redis_rq": not settings.redis_enabled},
        }
    )

    if settings.redis_enabled:
        try:
            from redis import Redis
            from rq import Queue

            queue = Queue(settings.rq_queue_name, connection=Redis.from_url(settings.redis_url or ""))
            queue.enqueue(
                execute_ingestion_job,
                job_id,
                scheme_id,
                filename,
                raw_bytes,
                source_url,
                source_type,
                demo_data,
            )
            log_event(logger, logging.INFO, "ingestion job enqueued", job_id=job_id, queue=settings.rq_queue_name)
            return IngestOutcome(job_id, "QUEUED", scheme_id, None, compute_content_hash(raw_bytes), 0, False, False)
        except Exception as exc:  # noqa: BLE001 - Redis down -> degrade to inline
            log_event(logger, logging.WARNING, "RQ enqueue failed; running ingestion inline", error=str(exc))

    log_event(logger, logging.INFO, "running ingestion inline (no Redis configured)", job_id=job_id)
    return await run_ingestion(
        job_id=job_id,
        scheme_id=scheme_id,
        filename=filename,
        raw_bytes=raw_bytes,
        source_url=source_url,
        source_type=source_type,
        demo_data=demo_data,
        db=db,
        settings=settings,
    )
