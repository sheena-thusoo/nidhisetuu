#!/usr/bin/env python
"""Ingest the demo sample documents into the RAG store (idempotent).

Idempotency: content-hash versioning means re-running this script produces the SAME
scheme version - chunks are upserted at their stable chunk IDs, never duplicated.

Usage:
    .venv/bin/python scripts/ingest_sample_docs.py            # all 4 sample docs
    .venv/bin/python scripts/ingest_sample_docs.py demo_term_loan   # one scheme

Runs inline (no Redis needed); set REDIS_URL to queue instead when using the API.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.services.ingestion import create_and_dispatch  # noqa: E402

SAMPLES = [
    ("demo_nfsdc_education_loan", "demo_nfsdc_education_loan.txt"),
    ("demo_nbcfdc_education_loan", "demo_nbcfdc_education_loan.txt"),
    ("demo_micro_finance", "demo_micro_finance.txt"),
    ("demo_term_loan", "demo_term_loan.html"),
]


async def main(argv: list[str]) -> int:
    settings = get_settings()
    wanted = set(argv) if argv else {scheme for scheme, _ in SAMPLES}
    failures = 0
    for scheme_id, filename in SAMPLES:
        if scheme_id not in wanted:
            continue
        path = settings.sample_docs_dir / filename
        if not path.exists():
            print(f"MISSING  {scheme_id}: {path} not found")
            failures += 1
            continue
        outcome = await create_and_dispatch(
            scheme_id=scheme_id,
            filename=filename,
            raw_bytes=path.read_bytes(),
            source_url=None,
            source_type="file",
            demo_data=True,
            settings=settings,
        )
        line = f"{outcome.status:<9} {scheme_id} v{outcome.version} chunks={outcome.chunks_indexed}"
        print(line + (f" error={outcome.error}" if outcome.error else ""))
        if outcome.status in {"FAILED", "SKIPPED"}:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise asyncio.run(main(sys.argv[1:]))
