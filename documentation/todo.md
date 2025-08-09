# TODO.md — cont3xt × Cline Integration (Code Gen Focus)

Goal: make cont3xt an MCP stdio server that Cline can use for **code generation tasks** with stable context packing, memory, and minimal deps. Ship a clean demo workflow.

---

## Milestones

**M1 — Cline MVP (today → +2 days)**

* [ ] Stable stdio server running locally with SQLite only
* [ ] Four tools: `health_ping`, `memory_upsert`, `search_memory`, `context_pack`
* [ ] Cline config checked in + working demo task
* [ ] Non-flaky token budgeting

**M2 — Quality & DX (week 1)**

* [ ] Unit + smoke tests (stdio launch, tool I/O)
* [ ] Typed, linted, and documented tool contracts
* [ ] Basic metrics & logs visible in terminal
* [ ] Minimal example repos + canned prompts

**M3 — Team-ready (week 2)**

* [ ] Optional retrieval backends gated behind env flags
* [ ] Docker image with stdio entrypoint
* [ ] CI: test, lint, type-check, build
* [ ] Versioned release `v0.1.1`

**M4 — Power user polish (stretch)**

* [ ] Dotted tool aliases (`context.pack`, etc.)
* [ ] Prompt cache & pack fingerprints
* [ ] Repo safety rails + dry-run planning mode

---

## 1) MCP Server API (hard requirements)

**Tool surfaces (final shape):**

* [ ] `health_ping() -> {"ok":true, "server":"cont3xt", "version":"0.1.1"}`
* [ ] `memory_upsert(session_id, content | user_input+assistant_response) -> {"ok":true,"ids":[...],"count":N,"meta":{...}}`
* [ ] `search_memory(session_id, query, max_results=10) -> {"results":[{id,text,score,created_at}], "meta":{...}}`
* [ ] `context_pack(session_id, current_input="", budget_tokens:int) -> {"pack": "…", "meta":{"input_tokens":X,"output_tokens":Y,"budget":B,"sources":[ids...]}}`

**Contracts**

* [ ] Always return envelope: `{"ok":bool, "data"|payload, "meta":{}}`
* [ ] Deterministic field names; no stack traces in responses
* [ ] Include `meta.duration_ms`, `meta.backend="sqlite"` for MVP
* [ ] Truncation policy documented (head/tail, separator)

---

## 2) Stdio Transport & Robustness

* [ ] `__main__.py` uses `mcp.server.stdio.stdio_server` and clean `server.run(read, write)`
* [ ] Handle EOF/pipe breaks → graceful exit code 0
* [ ] `--init-db` flag initializes SQLite and exits
* [ ] `PYTHONUNBUFFERED=1` in env for real-time logs

---

## 3) Persistence (SQLite-only MVP)

* [ ] `configs/*.yaml` supports `database.sqlite_path`
* [ ] Migrations: auto-create tables if missing
* [ ] Content hashing (optional) for idempotent upsert
* [ ] Simple substring + recency search as baseline
* [ ] Time/indexes added for `session_id`, `created_at`

---

## 4) Token Budgeting & Packing

* [ ] Implement `TokenCounter` (tiktoken) with model param
* [ ] Deterministic packing order: (current\_input) → recent hits → high-score hits
* [ ] Hard cap at `budget_tokens`; emit `BUDGET_TOO_SMALL` if nothing fits
* [ ] `meta.pack_stats` (num\_candidates, kept, dropped)

---

## 5) Observability

* [ ] Structured logs: tool name, args hash, duration, result size
* [ ] `--log-level` flag (info/debug)
* [ ] `health_ping` exposes version + backends

---

## 6) Cline Integration

**Repo config to check in: `.vscode/cline.json`**

```json
{
  "mcpServers": {
    "cont3xt": {
      "command": "virtual-context-mcp",
      "args": ["--config", "configs/novel_writing.yaml"],
      "env": { "CONTEXT_MAX_TOKENS": "12000", "PYTHONUNBUFFERED": "1" }
    }
  }
}
```

**Demo task (README snippet)**

* [ ] Add “Open Cline → Run Task: *Improve function X*”
* [ ] Task flow Cline should follow:

  1. Call `search_memory(session_id, query="project goals")`
  2. Call `context_pack(session_id, current_input="<file diff request>", budget_tokens=12000)`
  3. Generate plan → apply small diff → run tests
  4. Call `memory_upsert` with (user\_input, assistant\_response) to persist the session outcome

**Acceptance**

* [ ] From a clean clone, user can run `pip install -e .`, launch Cline, and complete a guided code edit PR in one go.

---

## 7) Safety Rails for Code Gen

* [ ] “Planning Mode” tool response with file list + diffs preview, require user confirmation (Cline UX already supports)
* [ ] Deny edits outside repo root (`./`) and git-ignored paths unless `ALLOW_OUTSIDE=1`
* [ ] Optional `DRY_RUN=1` env → only propose diffs, no file writes

---

## 8) Developer Experience

* [ ] `Makefile` (or `justfile`):

  * `make dev` → install deps
  * `make run` → stdio server
  * `make initdb` → create DB
  * `make test` → run tests
  * `make lint`, `make type`
* [ ] `requirements.txt` minimal; heavy deps moved to extras:

  * `pip install cont3xt[spacy,qdrant,neo4j]` (later)
* [ ] Pre-commit hooks (black/ruff/mypy)

---

## 9) Tests

**Unit**

* [ ] Token budgeter trims correctly
* [ ] `memory_upsert` returns IDs and count
* [ ] `search_memory` on empty DB returns `[]`
* [ ] Error mapping: DB locked / bad args

**Smoke**

* [ ] Launch stdio, call each tool, assert envelopes
* [ ] Kill stdin; process exits cleanly in <1s

**Perf (light)**

* [ ] 1k memory rows: `search_memory` < 50ms median
* [ ] `context_pack` 12k tokens in < 300ms median on laptop CPU

---

## 10) Docker & CI

**Docker**

* [ ] `docker/Dockerfile` runs: `CMD ["virtual-context-mcp","--config","configs/novel_writing.yaml"]`
* [ ] Mount `./data` for SQLite persistence
* [ ] Multi-arch build (optional)

**CI (GitHub Actions)**

* [ ] `python -m pip install -e .[test]`
* [ ] Run `ruff`, `mypy`, `pytest`
* [ ] Build Docker (smoke run `--init-db`)

---

## 11) Docs & Examples

* [ ] Update `README.md` with:

  * Quickstart (local + Docker)
  * Cline config
  * Tool reference (args, responses)
  * Demo workflow (improve function, add unit test, run)
* [ ] `examples/`:

  * `demo_repo/` tiny Python lib + failing test
  * `scripts/seed_memory.py` to pre-fill session knowledge

---

## 12) Versioning & Release

* [ ] Bump to `0.1.1` (fits the “11” vibe)
* [ ] Tag + release notes (what changed vs story tools)
* [ ] Changelog with API contracts

---

## Stretch / Nice-to-haves

* [ ] Dotted aliases (`context.pack`, `search.memory`, `memory.upsert`)
* [ ] BM25 or Lite vector search behind `CTX_ENABLE_*`
* [ ] Pack fingerprints → skip recompute if unchanged
* [ ] Repo-level `cont3xt.yaml` for project-specific memory/session IDs
* [ ] Simple “skills” memory lane: store/refactor prompts that worked well

---

## Open Questions

* [ ] Do we enforce idempotency by hash on `memory_upsert`?
* [ ] Include file paths in `search_memory` results (when source-known)?
* [ ] Add `session_tags` (e.g., `["codegen","refactor"]`) for filtering?

---

### Command quickies

```bash
# Dev
pip install -e .
virtual-context-mcp --init-db
virtual-context-mcp --config configs/novel_writing.yaml

# Tests
pytest -q
ruff check .
mypy src
```

---

When M1 is done, ping me and I’ll sanity-check the Cline demo flow and tighten any rough edges before we cut `v0.1.1`.
