# NidhiSetu — Final Report

Portfolio project: an evidence-backed agentic decision-support API for Indian government
loan/education schemes. **8 phases, fully autonomous build.** Everything below is backed by
unedited command output in [BUILD_LOG.md](BUILD_LOG.md).

## Headline result

```
.venv/bin/python -m pytest tests -q
201 passed, 2 skipped in 3.44s
```

The 2 skips are the **live** integration tests (`live_tavily`, `live_redis`) — correctly
SKIPPED, never counted as passed, because no third-party credentials exist in the build
environment. Every component that needs a key ran in **MOCK MODE** with greppable
`[MOCK MODE]` logs.

```
NidhiSetu eval: 9/9 cases passed        # scripts/evaluate_demo.py, deterministic golden set
mock_mode: {'groq': True, 'pinecone': True, 'tavily': True, 'redis_rq': True}
```

## Phase-by-phase (real test counts)

| Phase | Scope | Result |
| --- | --- | --- |
| 0 | venv (Python 3.11.16 via uv), pinned requirements | installed OK |
| 1 | Mongo layer, rule engine (rural/urban), EMI engine, partners | 61 → 72 passed |
| 2 | LangGraph flows, tool gateway, intake/explanation agents, RAG + lexical evidence | 129 passed |
| 3 | Chunker/parsing quality + retrieval service | 151 passed |
| 4 | Ingestion pipeline + admin API + FastAPI app shell | 171 passed, 1 skipped |
| 5 | Tavily client (mock+live) + `/schemes/match`, `/research` | 184 passed, 2 skipped |
| 6 | JWT auth + 5-state application lifecycle | 196 passed, 2 skipped |
| 7 | slowapi rate limiting + deterministic eval harness | 201 passed, 2 skipped |
| 8 | README, env template, this report | — |

## Mock vs real status

| Component | Status in this build | What a key would change |
| --- | --- | --- |
| Rule/EMI/eligibility engines | **REAL** (pure Python, no external dependency) | nothing |
| MongoDB (Motor) | REAL API against **mongomock-motor** in tests; real Mongo expected at runtime | point `MONGODB_URI` at a server |
| Groq LLM | MOCK: regex intake, template explanations, lexical verification | LLM fills intake gaps, writes explanations (numeric-guarded), classifies evidence |
| Pinecone | MOCK: in-memory cosine store + hash embeddings | hosted index + `multilingual-e5-large` semantic embeddings |
| Tavily | MOCK: deterministic labelled placeholders (live path unit-tested with a fake HTTP transport) | real web research, always labelled external |
| Redis/RQ | MOCK: inline job execution (same pipeline + job records) | background ingestion workers |

## What the system does (verified by tests, not assertions)

- Free-text or structured applicant facts → per-scheme **three-state eligibility**
  (`potentially_eligible` / `insufficient_data` / `not_eligible`) with per-rule pass/fail
  reasons and exact rural/urban boundary behaviour (81,000 passes; 81,001 fails).
- Deterministic **EMI/totals** for candidate schemes (80,000 @ 10% / 60m → 1,699.76, 2-dp
  Decimal, `ROUND_HALF_UP`) with typed validation errors.
- **RAG ingestion** (.txt + table-aware .html) with content-hash versioning: same content is
  idempotent, changed content bumps the version and supersedes old vectors; verified citations
  carry scheme/version/section/URL/freshness provenance.
- **Evidence verification** of every claim with four statuses and a deterministic conflict
  detector that the LLM can never override; LLM "support" without a citation is downgraded.
- **External research** (Tavily) always labelled `external_research`, never merged with
  internal evidence; used in match mode only as an explicit fallback when internal evidence
  is insufficient.
- **JWT auth** (bcrypt, HS256) and a **five-state application lifecycle**
  (DRAFT → SUBMITTED → UNDER_REVIEW → APPROVED/REJECTED) with ownership checks, an optimistic
  status guard, and an append-only audit trail.
- **Ops surface**: request-id middleware (echoed to clients), key-only tool logging, slowapi
  limits (30/15/20 per minute), `/health` reporting the mock-mode matrix, `/api/tools` catalog.

## Design decisions (and why)

1. **Deterministic core / agentic shell.** The CORE RULE (LLM never computes) is enforced
   structurally: money and eligibility flow only through gateway tools marked
   `never_llm=True`, and the explanation agent runs a numeric guard that discards any LLM
   text containing numbers absent from the deterministic payload.
2. **Three-state eligibility.** A missing critical field must never silently read as "fails"
   or "passes" — `insufficient_data` makes missing data visible and drives the
   `unresolved_questions` UX.
3. **Content-addressed chunk IDs + version supersession.** Ingestion is idempotent by
   construction; retrieval always reflects the current document version while Mongo keeps the
   full version history for audit.
4. **Fixed lexical verification thresholds** (SUPPORTED ≥ 0.25 overlap with all numbers,
   CONFLICT ≥ 0.5 overlap without the numbers, no-number claims ≥ 0.5/0.25). Deterministic
   verdicts always run first; the LLM can only confirm or upgrade — never fabricate citations.
5. **Mock mode as a first-class mode**, not an afterthought: every missing key degrades to a
   clearly-logged deterministic behaviour, so the whole product (API, graph, jobs, eval) is
   demonstrable and testable offline.
6. **Gateway in the middle.** Every agent→tool call gets Pydantic in/out validation, hard
   timeouts, typed error envelopes, and key-only logging — one place to reason about failure
   and privacy.

## Bugs found and fixed during verification (honest list)

1. LangGraph node/state key collision (`external_research`, `explanation` used as both node
   and state keys) crashed graph compilation — nodes renamed (Phase 2).
2. A failed `rag_search` tool call was misreported as "no indexed chunks" — distinct failure
   warnings added (Phase 2).
3. Chunker labelled chunks with the NEXT section's heading when emitting at a section change
   (Phase 3).
4. `is_table_row` was effectively dead (only the first block set it) — chunks containing
   table rows are now always flagged (Phase 3).
5. Char-based overlap split table rows mid-line across chunks — overlap tails now snap to
   line boundaries (Phase 3).
6. Version bumps left old-version chunks in the vector store, mixing v1/v2 evidence — old
   versions are now superseded (Phase 4).
7. Job documents' `mock_mode` flag map was overwritten by a bare bool at completion (Phase 4).
8. mongomock/Mongo `_id` (ObjectId) leaked through the auth service into response
   serialization and crashed it (Phase 6).
9. slowapi requires the endpoint parameter to be *named* `request`; more subtly,
   `from __future__ import annotations` + slowapi's wrapper made FastAPI resolve string
   annotations against slowapi's module namespace, silently turning body params into query
   params (all POSTs 422). Fixed and documented in a module docstring so it isn't "cleaned
   up" back into a bug (Phase 7).
10. Three initial eval "failures" were golden-set authoring errors (age phrases the regex
    intentionally requires were missing) — the extractor was NOT loosened to pass (Phase 7).

## Limitations / underclaiming

- **No live third-party call was ever executed** during this build (no keys). The live-path
  client code is covered by fake-transport tests; the real-network tests are the two SKIPPED
  live markers. Claims about live behaviour are therefore structural, not observational.
- Mock retrieval quality is **lexical/numeric overlap**, not semantic; no retrieval-quality
  claims are made beyond the in-process tests.
- All scheme rules and partner records are clearly-labelled **demo placeholders**.
- The lexical conflict detector can flag some rural/urban sentence pairs inside one document
  as CONFLICTING (documented deterministic rule) — semantic conflict detection is a live-LLM
  upgrade path, not implemented here.
- Admin surface uses a static `ADMIN_API_KEY` header (not RBAC); auth has no refresh tokens;
  bcrypt rounds=10 (test speed).
- The eval harness is intentionally small (9 deterministic cases) — it validates routing and
  boundary behaviour, not model quality.

## Go-live checklist (add keys, in this order)

1. `MONGODB_URI` + `MONGODB_DB` — real Mongo; run `.venv/bin/python scripts/ingest_sample_docs.py`.
2. `JWT_SECRET` + `ADMIN_API_KEY` — real secrets (32+ random chars).
3. `GROQ_API_KEY` — enables LLM intake fill, LLM explanations (numeric-guarded), LLM evidence
   classification (lexical verdicts still win conflicts).
4. `PINECONE_API_KEY` — hosted index + real embeddings; re-ingest documents once so vectors
   are (re)created with the semantic model.
5. `TAVILY_API_KEY` — real external research (still quarantined from internal evidence).
6. `REDIS_URL` — ingestion moves to background workers (`rq worker ingestion`).
7. Set `RATE_LIMIT_ENABLED=true` and review the per-endpoint limits.
8. Re-run `.venv/bin/python -m pytest tests -q` — the two live tests must now RUN (not skip).

Environment template: [`env.example`](env.example). Per-phase evidence: [`BUILD_LOG.md`](BUILD_LOG.md).
