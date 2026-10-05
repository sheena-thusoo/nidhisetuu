# NidhiSetu

**Evidence-backed agentic decision-support API for Indian government loan/education schemes.**
FastAPI + Python 3.11. Portfolio project: fully functional offline in MOCK MODE, upgraded by
live keys when provided.

> **CORE RULE — the LLM never computes money or eligibility.** Every EMI, interest figure,
> threshold comparison and eligibility outcome comes from deterministic Python services
> (`app/services`, `app/rules`, Decimal-exact). The agentic layer (LangGraph + Groq Llama 3.3)
> only interprets free text, retrieves evidence, classifies/verifies claims, explains and routes.
> A numeric guard discards any LLM explanation containing a number that is not in the
> deterministic payload.

---

## Architecture

```
HTTP (FastAPI) ──> LangGraph flows ──> ToolGateway ──> tools ──> deterministic services
   |                  |                  (validation,
   |                  |                   timeouts, error
   |                  |                   envelopes, key-only
   |                  |                   logging, trace)
   |                  |
   |                  ├─ match flow:   intake → eligibility(+financials, partners)
   |                  │                → RAG retrieval → evidence verification
   |                  │                → external research ONLY as labelled fallback
   |                  │                → explanation
   |                  └─ research flow: intake → RAG retrieval → evidence verification
   |                                   → external research (ALWAYS, labelled separately)
   │                                   → explanation
   ├──> MongoDB (Motor, async): users, schemes, scheme_versions, documents,
   │    ingestion_jobs, applications, audit_events
   ├──> Vector store: Pinecone (live) or in-memory cosine store (mock)
   ├──> Embeddings: Pinecone inference (live) or deterministic hash vectorizer (mock)
   ├──> Jobs: RQ + Redis (live) or inline execution (mock)
   └──> External research: Tavily (live) or labelled placeholder items (mock)
```

**Deterministic core (never an LLM):**

| Module | Responsibility |
| --- | --- |
| `app/services/financials.py` | Pure EMI engine: Decimal, `ROUND_HALF_UP`, 2-dp money, typed error codes, amortisation |
| `app/services/eligibility.py` | Scheme evaluation, 3-state eligibility, financials assembly |
| `app/rules/operators.py` + `engine.py` | Data-driven rule operators (`in`, `between`, `gte`, `eq`, `threshold_map`…) |
| `app/rules/loader.py` | Scheme rule JSON loading/validation (`RuleDataError` on bad data) |
| `app/services/partners.py` | Deterministic partner matching over clearly-labelled demo data |
| `app/services/freshness.py` | FRESH ≤ 180d / AGING ≤ 365d / STALE classification |
| `app/services/ingestion.py` | parse → hash → version → chunk → embed → upsert, with job records |

**Agentic layer (`app/ai`, `app/rag`, `app/research`, `app/tools`):** intake extraction
(regex always; LLM fills gaps in live mode), explanation with numeric guard, RAG chunking /
retrieval, lexical + optional-LLM evidence verification, Tavily research client, and the
`ToolGateway` every node must call.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | – | Liveness + which components are in mock mode |
| GET | `/api/tools` | – | Tool catalog the agent uses (deterministic tools flagged) |
| POST | `/schemes/match` | – | Applicant query/facts → eligibility, financials, partners, evidence verdicts, explanation |
| POST | `/research` | – | Query → verified internal chunks + separately-labelled external results |
| POST | `/admin/ingest` | `X-Admin-Key` | Ingest a scheme document (.txt/.html, table-aware) |
| GET | `/admin/ingest/{job_id}` | `X-Admin-Key` | Job status (QUEUED/RUNNING/COMPLETED/FAILED/SKIPPED) |
| GET | `/admin/jobs` | `X-Admin-Key` | Recent ingestion jobs |
| POST | `/auth/register` `/auth/login` | – | JWT auth (bcrypt); admin users via `admin_invite_code == ADMIN_API_KEY` |
| GET | `/auth/me` | Bearer | Current user |
| POST | `/applications` | Bearer | Create application (with deterministic eligibility snapshot) |
| GET | `/applications`, `/applications/{id}`, `/applications/{id}/audit` | Bearer | Owner-scoped reads (admins see all) |
| POST | `/applications/{id}/submit` | Bearer (owner) | DRAFT → SUBMITTED |
| POST | `/applications/{id}/review` | Bearer (admin) | SUBMITTED → UNDER_REVIEW |
| POST | `/applications/{id}/decision` | Bearer (admin) | UNDER_REVIEW → APPROVED / REJECTED |

Rate limits (slowapi): match 30/min, research 15/min, auth 20/min, keyed by IP
(disable with `RATE_LIMIT_ENABLED=false`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Start Mongo (or point MONGODB_URI at an existing instance)
# 2. Ingest the demo documents (idempotent - safe to re-run)
.venv/bin/python scripts/ingest_sample_docs.py

# 3. Run the deterministic eval (9 golden cases, offline)
.venv/bin/python scripts/evaluate_demo.py

# 4. Serve the API
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Copy `env.example` to `.env` and add keys as available — the API runs without any of them.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

201 passed, 2 skipped (expected in mock mode):
`live_tavily` and `live_redis` are real-integration tests that **SKIP** (never pass) unless
their credentials exist. Real per-phase output is recorded in [BUILD_LOG.md](BUILD_LOG.md);
honest phase-by-phase summary in [FINAL_REPORT.md](FINAL_REPORT.md).

Mongo is faked with `mongomock-motor` (same async Motor API), the vector store/embeddings run
in-process, and jobs execute inline — no external services needed for the suite.

## Mock mode matrix

| Component | Missing key | Mock behaviour (always logged `[MOCK MODE]`) |
| --- | --- | --- |
| Groq LLM | `GROQ_API_KEY` | regex-only intake, template explanations, lexical-only verification |
| Pinecone | `PINECONE_API_KEY` | in-memory cosine store + deterministic 256-dim hash embeddings |
| Tavily | `TAVILY_API_KEY` | deterministic placeholder items, labelled `external_research`, never merged with internal evidence |
| Redis/RQ | `REDIS_URL` | ingestion runs inline; identical job records |

## Design decisions

1. **LLM never computes** — the rule/financial engines own all math; the LLM only interprets,
   retrieves, classifies and explains. Verified structurally: agentic nodes reach money only
   through gateway tools marked `never_llm=True`.
2. **Three-state eligibility** — `not_eligible` if ANY critical rule fails; `insufficient_data`
   if a critical field is missing (and nothing failed); else `potentially_eligible`.
   Non-critical rules are advisory and never change the status.
3. **`threshold_map` operator** — rural/urban income limits read from `location_type` with a
   `default` fallback (and single-variant fallback), so missing location never crashes.
4. **Stable, content-addressed chunk IDs** — `sha256(scheme_id|v{version}|content_hash|chunk_index)[:32]`;
   re-ingesting the same content overwrites instead of duplicating. A changed document bumps the
   version and supersedes the previous version's vectors (history kept in Mongo).
5. **Freshness on every citation** — FRESH ≤ 180d, AGING ≤ 365d, STALE beyond; every evidence
   source carries `retrieved_at` + classification.
6. **Hash embeddings in mock mode** — deterministic, byte-identical for identical input, numbers
   weighted ×3. Explicitly NOT a semantic model: retrieval quality in mock mode is
   lexical/numeric overlap only (underclaimed in reports).
7. **Lexical evidence verification with fixed thresholds** — SUPPORTED needs every claim number
   present and token overlap ≥ 0.25; CONFLICTING when a ≥ 0.5-overlap chunk lacks the claim's
   numbers; number-less claims need ≥ 0.5 (SUPPORTED) / ≥ 0.25 (PARTIAL). A deterministic
   CONFLICT always beats the LLM; an LLM "SUPPORTED" without a deterministic citation is
   downgraded to INSUFFICIENT_EVIDENCE — citations are never fabricated.
8. **External research is quarantined** — always labelled `external_research` with a caveat,
   reported separately from internal evidence, used in match mode only as an explicitly
   permitted fallback (`allow_external_fallback`).
9. **Tool gateway guarantees** — Pydantic input AND output validation, hard asyncio timeouts,
   typed error envelopes (`unknown_tool`, `invalid_input`, `timeout`, `invalid_output`, …),
   input KEYS logged but never values, a full per-request tool trace.
10. **Table-aware HTML ingestion** — `<tr>` rows render as single pipe-delimited lines the
    chunker keeps intact and flags `is_table_row=True`; a plain `get_text()` path would
    flatten cell structure (documented in `app/rag/parsing.py`).
11. **Privacy by default** — e-mails masked, sensitive keys redacted, request bodies never
    logged; every log line carries the `X-Request-ID`.

## Known limitations (read before trusting it)

- **All scheme data is demo data** (`demo_data: true`, `demo_placeholder` sources) — replace via
  `/admin/ingest` with official documents before any real use.
- **No live third-party path was executed during the build** (no keys in the environment);
  live Groq/Pinecone/Tavily/Redis behaviour is covered by marked tests that SKIP, plus
  fake-transport tests for the client mapping code.
- The lexical conflict detector can mark some rural/urban sentence pairs in the same document
  CONFLICTING (documented rule: claim numbers absent from a ≥ 0.5-overlap chunk); semantic
  conflict detection is a live-LLM upgrade path.
- Admin auth is a static `ADMIN_API_KEY` header (portfolio-appropriate, not RBAC).
- Auth uses a single 24h JWT (no refresh tokens); bcrypt rounds=10 (speed over paranoia).
- The eval harness is deliberately small (9 golden cases) — it verifies deterministic routing,
  not model quality.
