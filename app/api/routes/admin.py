"""Admin API: document ingestion (Phase 4) + ingestion job status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.deps import AdminDep, DbDep, SettingsDep
from app.services.ingestion import create_and_dispatch
from app.utils.logging import log_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    _admin: AdminDep,
    scheme_id: str = Form(...),
    file: UploadFile = File(...),
    source_url: str | None = Form(None),
    demo_data: bool = Form(True),
    db: DbDep = None,
    settings: SettingsDep = None,
):
    """Ingest a scheme guideline document (.txt or .html) for a known scheme_id.

    With Redis configured the job is queued (202 + QUEUED); without Redis it runs
    inline and the response already carries the final result (still 202 because the
    API contract is asynchronous).
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file exceeds 10 MB limit")
    filename = file.filename or "document.txt"

    outcome = await create_and_dispatch(
        scheme_id=scheme_id,
        filename=filename,
        raw_bytes=raw,
        source_url=source_url,
        source_type="file",
        demo_data=demo_data,
        db=db,
        settings=settings,
    )
    log_event(
        logger, logging.INFO, "ingest request handled",
        scheme_id=scheme_id, job_id=outcome.job_id, status=outcome.status,
    )
    return {
        "job_id": outcome.job_id,
        "status": outcome.status,
        "scheme_id": outcome.scheme_id,
        "version": outcome.version,
        "content_hash": outcome.content_hash,
        "chunks_indexed": outcome.chunks_indexed,
        "version_bumped": outcome.version_bumped,
        "executed_inline": outcome.executed_inline,
        "error": outcome.error,
        "_links": {"job_status": f"/admin/ingest/{outcome.job_id}"},
    }


@router.get("/ingest/{job_id}")
async def job_status(job_id: str, _admin: AdminDep, db: DbDep = None):
    job = await db.ingestion_jobs.find_one({"job_id": job_id}, {"_id": 0})
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown job {job_id!r}")
    return job


@router.get("/jobs")
async def recent_jobs(_admin: AdminDep, db: DbDep = None, limit: int = 20):
    limit = max(1, min(limit, 100))
    cursor = db.ingestion_jobs.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    return {"jobs": [doc async for doc in cursor]}
