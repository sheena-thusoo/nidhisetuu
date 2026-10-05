# BUILD_LOG

Real, unedited results of every phase verification run (`pytest -q`, from the project root).
Environment: Python 3.11.16 (uv-managed venv at `.venv/`), Ubuntu 22.04 sandbox.
MongoDB for tests: `mongomock-motor` in-memory (no server required).
No live third-party keys were present in the build environment, so Groq / Pinecone / Tavily / Redis
paths ran in **MOCK MODE** and their live-integration tests are SKIPPED (never reported as passed).

---

## Phase 1 - MongoDB + rule engine (rural/urban) + EMI engine + partner service

Command:
```
.venv/bin/python -m pytest tests/unit -q
```

Result:
```
.............................................................            [100%]
61 passed in 0.09s
```

Notes:
- Includes rural/urban income boundary tests (81,000 / 81,001 / 90,000 / 1,03,000 / 1,03,001, missing location -> default threshold, missing income -> `insufficient_data`).
- EMI engine: known-value (100,000 @ 10% / 12m = 8,791.59), zero-interest, 1-month and 360-month tenures, 2-dp quantization, typed validation errors.
- Two initial boundary failures were in the *test* (microsecond drift made 180 days into 180.0000001); tests now pin `now` explicitly. No production code was changed to make them pass.

Re-run after Phase 2 (same command, `tests/unit` also now contains the db-layer tests):
```
72 passed in 0.11s
```

---

## Phase 2 - Agentic layer: LangGraph flows, tool gateway, intake/explanation agents, RAG retrieval + lexical evidence verification

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
........................................................................ [ 55%]
.........................................................                [100%]
129 passed in 0.89s
```

Breakdown: 72 pre-existing unit tests + 57 new tests in `tests/ai/` (intake regex + merge precedence + numeric guard, gateway validation/timeout/unknown-tool/key-only logging, graph match & research flows) and `tests/rag/` (lexical evidence statuses incl. CONFLICTING precedence, claim building from evaluations, LLM-merge rules with fakes).

MOCK MODE behaviour verified in-run (all via `[MOCK MODE]` log lines, greppable):
- Groq absent -> regex-only intake, template explanation, lexical-only verification.
- Pinecone absent -> in-memory vector store + deterministic hash embeddings (256 dims).
- Tavily absent -> `external_research` tool unregistered at Phase 2 (registered in Phase 5); the graph degrades with a warning and `external_mock_mode=True`, it never blocks the flow.

Bugs found and fixed during Phase 2 verification (with regression tests now in place):
1. `build_graph` crashed: LangGraph node names `external_research` / `explanation` collided with same-named `GraphState` keys (`'external_research' is already being used as a state key`). Nodes renamed to `external_research_node` / `explain`; routing tables updated.
2. Product gap: when the `rag_search` tool FAILED (e.g. store down), the retrieval node reported the misleading warning "no indexed document chunks"; it now emits `rag_search failed: <reason>` per scheme (and in research mode). New test `test_rag_search_failure_degrades_with_warning` covers it.
3. Test double bug (found by a failing assertion, not silently): the fake LLM keyed replies on claim IDs that never appear in the LLM prompt, so merge-path tests initially passed vacuously; fakes now key on claim text and the merge assertions are exact.

Underclaiming notes (what is NOT yet verified at Phase 2):
- Live Groq / Pinecone / Tavily paths: never executed (no keys); live tests are marked live_* and SKIP.
- Retrieval quality is lexical/numeric-overlap only (hash embeddings); no semantic ranking claim is made.
- The explanation agent's live LLM text and the numeric guard against a real LLM are only exercised with fakes.

---

## Phase 3 - RAG chunking/parsing quality + retrieval service

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
........................................................................ [ 95%]
.......                                                                  [100%]
151 passed in 0.83s
```

New: 22 tests in `tests/rag/` (chunker determinism/heading preservation/table grouping/stable IDs; HTML table-aware parsing; RagService index/search/filter/freshness/idempotent re-ingestion; hash-embedding determinism).

New module: `app/rag/parsing.py` - table-aware document parser. HTML `<tr>` rows are rendered as ONE pipe-delimited line each ("| Loan slab | 13.5 | 24 - 36 |"), which the chunker keeps together and flags `is_table_row=True`. A plain `BeautifulSoup.get_text()` path was rejected because it flattens table cells and loses row grouping (documented in the module docstring). Verified against `data/sample_documents/demo_term_loan.html`: 4 pipe lines (header + 3 slabs) survive chunking intact.

Bugs found and fixed during Phase 3 verification:
1. Chunker section mislabeling: a chunk emitted at a section change was labelled with the NEXT section's heading (the label was switched before emit). Emit now happens first; the label switch after.
2. `is_table_row` was effectively dead: it only reflected the first block, so a heading+table chunk was never flagged. A chunk containing table rows is now always flagged.
3. Char-based overlap could start a chunk mid-line, splitting a table row across chunks ("... | 84 |" fragment observed in the smoke run). Overlap tails now snap to the next line boundary.

Limitation (by design, to document): `.pdf`/`.docx` raise `UnsupportedDocumentType`; ingestion records this in the job rather than guessing. PDF support via pdfplumber is a future extension.

---

## Phase 4 - Ingestion pipeline (parse -> hash -> version -> chunk -> embed -> upsert) + admin API + FastAPI app shell

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
........................................................................ [ 83%]
............................                                             [100%]
171 passed, 1 skipped in 1.27s
```

The single SKIP is `live_redis` (RQ enqueue against a live Redis) - correctly skipped, never counted as a pass, because REDIS_URL is intentionally unset in this environment.

New: `app/services/ingestion.py` (pipeline + job records + RQ enqueue with inline fallback), `app/main.py` (app factory with lifespan), `app/api/` (request-id middleware, admin-key dependency, `/admin/ingest` + `/admin/ingest/{job_id}` + `/admin/jobs`, `/health` with mock-mode flags, `/api/tools` catalog), `scripts/ingest_sample_docs.py` (idempotent; verified twice against the same database: identical versions both runs, vector-store count unchanged at 14 chunks).

Behaviour verified by tests:
- same content -> same version (idempotent), changed content -> version+1 with the previous version's chunks superseded out of the vector store (history kept in Mongo `scheme_versions`/`documents`);
- unknown `scheme_id` -> job FAILED (HTTP still 202; the failure is in the job record), HTML `<meta name="scheme_id">` mismatch -> FAILED, empty document -> SKIPPED;
- admin API: 401 without key, 403 with wrong key, 400 empty file; `X-Request-ID` middleware echoes a provided id and generates one otherwise.

Bugs/gaps found and fixed during Phase 4 verification:
1. Version-bump left the previous version's chunks in the vector store, so retrieval mixed v1 and v2 evidence. Old version is now superseded on bump.
2. The job document's `mock_mode` field was overwritten from a flag map at enqueue to a bare bool at completion; it is now a stable component map `{redis_rq, vector_store, llm}` plus `executed_mode` (`inline` | `rq_queue`).
3. Integration-suite cross-test pollution: the RAG service singleton leaked chunks between tests; the integration conftest now injects a fresh in-memory service per test (product code unchanged for this one).

---

## Phase 5 - External research (Tavily) + agentic HTTP endpoints (/schemes/match, /research)

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
.........................................s.............................. [ 77%]
..........................................                               [100%]
184 passed, 2 skipped in 2.04s
```

The 2 SKIPS are `live_redis` and `live_tavily` - credentials intentionally absent in this environment; they can never silently pass.

New: `app/research/tavily_client.py` (deterministic mock items when TAVILY_API_KEY is absent; live httpx path mapped 1:1 onto `ExternalResearchItem`; `TavilyError` on failure), the `external_research` gateway tool is now REGISTERED (8 tools total) and used by both flows, `POST /schemes/match` and `POST /research` endpoints with request/response models in `app/schemas/api.py`.

Behaviour verified by tests:
- mock Tavily items are deterministic, always labelled `external_research`, always caveated, never merged with internal evidence (internal chunk ids and external urls asserted disjoint);
- match flow: structured facts and pure free-text intake both reach the same deterministic evaluation; 400 when neither is provided; 422 on out-of-range top_k;
- research flow: empty store -> explicit "no chunks" warning (honest degradation), ingested store -> verified chunk rows with freshness + provenance;
- `X-Request-ID` middleware id is reused by the routes (response body `request_id` == response header).

Notes / known limitations (documented, not hidden):
- The demo NFSDC document contains both a rural (81,000) and an urban (1,03,000) income sentence, so the lexical conflict detector legitimately marks some sentence-pairs CONFLICTING_EVIDENCE in research mode. That is the documented deterministic rule (claim numbers absent from a >=0.5-overlap chunk), not a bug; semantic conflict detection is a live-LLM upgrade path.
- Live Tavily behaviour is verified only against a fake HTTP transport; the real-API test is the skipped live_tavily marker.

---

## Phase 6 - JWT auth + five-state application lifecycle

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
..........s..........................................s.................. [ 72%]
......................................................                   [100%]
196 passed, 2 skipped in 3.20s
```

New: `app/services/auth.py` (bcrypt hashing rounds=10, HS256 tokens via PyJWT, typed `AuthError`), `app/services/applications.py` (DRAFT -> SUBMITTED -> UNDER_REVIEW -> APPROVED/REJECTED with an explicit transition table, ownership checks, optimistic status guard, append-only audit trail in `audit_events`), `/auth/register|login|me` and `/applications*` routes, `get_current_user`/`AdminUser` dependencies. Admin users are created only with `admin_invite_code == ADMIN_API_KEY`.

Behaviour verified by tests:
- register -> login -> /auth/me round-trip; duplicate email 409; short password 422; bad invite code 403; wrong password 401; tampered/missing token 401; password hashes never leave the service;
- full lifecycle incl. deterministic `eligibility_snapshot` (status + `computed_by: app/services/eligibility.py`) captured at creation; invalid transitions (approve a DRAFT, double submit) 409; ownership enforced with 404 (existence not revealed to strangers); non-admin review 403; admin can read any application; audit trail shows the exact status chain.

Bug fixed during Phase 6 verification:
1. mongomock (like real Mongo insert paths) attaches `_id` (ObjectId) to inserted/returned documents; the auth service returned it, which crashed response serialization (`PydanticSerializationError: unknown type ObjectId`). All user-returning paths now strip `_id` and `password_hash`.
2. Added pinned dependency `email-validator==2.2.0` (required by pydantic `EmailStr`).

---

## Phase 7 - Rate limiting (slowapi) + deterministic eval harness

Command:
```
.venv/bin/python -m pytest tests -q
```

Result:
```
........................................................................ [ 35%]
............s.............................................s............. [ 70%]
...........................................................              [100%]
201 passed, 2 skipped in 3.44s
```

New: `app/api/ratelimit.py` (slowapi limiter keyed by client IP; enabled from settings; `SlowAPIMiddleware` + 429 handler registered in the app factory) applied to `/schemes/match` (30/min), `/research` (15/min), `/auth/register|login` (20/min); `scripts/evaluate_demo.py` - a 9-case deterministic golden set (rural/urban boundaries, category gates, insufficient-data, underage, structured facts) run through the real match flow with `allow_external_fallback=False`; the harness compares pipeline output to expected statuses and computes nothing itself.

Eval result (real run):
```
NidhiSetu eval: 9/9 cases passed
mock_mode: {'groq': True, 'pinecone': True, 'tavily': True, 'redis_rq': True}
```

Bugs found and fixed during Phase 7 verification:
1. slowapi's `limit` decorator requires the endpoint parameter to be literally named `request` (name-based lookup) - payload params renamed accordingly.
2. REAL TRAP, worth documenting: combining `from __future__ import annotations` (PEP 563) with slowapi's wrapper broke FastAPI's type resolution - FastAPI evaluated the string annotations against the WRAPPER's module globals (slowapi's), silently demoting the typed body parameter to a required query param (all POSTs became 422). Fixed by dropping the postponed-annotations import in the two decorated route modules; documented in a module docstring note so it is not "cleaned up" back into a bug later.
3. Three initial eval "failures" were golden-set authoring errors (age phrases the regex extractor intentionally requires were absent), not engine bugs; the extractor was NOT loosened to make the eval pass.

---

## Phase 8 - Documentation + final report

Commands (final verification):
```
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/evaluate_demo.py
.venv/bin/python -c "from app.main import create_app; app = create_app(); print('routes:', len(app.routes))"
```

Results:
```
201 passed, 2 skipped in 3.57s
NidhiSetu eval: 9/9 cases passed
mock_mode: {'groq': True, 'pinecone': True, 'tavily': True, 'redis_rq': True}
routes: 21
```

New: `README.md` (architecture, endpoint table, quick start, mock-mode matrix, 11 design
decisions, limitations), `env.example` (every setting in `app/config.py`; named without the
leading dot because the workspace blocks writing `.env*` files - copy to `.env`), and
`FINAL_REPORT.md` (phase-by-phase real results, mock-vs-real matrix, honest bug list, go-live
checklist). Also fixed the stale `APPLICATION_STATUSES` constant in `app/constants.py` to
match the implemented lifecycle (DRAFT/SUBMITTED/UNDER_REVIEW/APPROVED/REJECTED).

---

## End state

- 201 tests passed, 2 live-integration tests SKIPPED (credentials intentionally absent; live
  tests are built and marked `live_tavily` / `live_redis`, never counted as passes).
- Deterministic eval: 9/9 golden cases pass offline.
- All third-party components ran in MOCK MODE throughout the build with greppable
  `[MOCK MODE]` logs; go-live steps are listed in FINAL_REPORT.md.
