Integrated cont3xt into Cline per the checkpoint and TODO, completing M1.

Implemented changes
- Standardized tool envelopes
  - All tools now return: {"ok": boolean, "data": {...}, "meta": {"duration_ms": number, "backend": "sqlite", "version": "0.1.1"}}
  - Errors: ok=false with meta.error {code, message}; no stack traces.
  - Added timing via perf_counter.
- Token budgeting and context pack
  - context_pack now includes pack.pack_stats {num_candidates, kept, dropped, buffer_tokens}.
  - Emits "reason": "BUDGET_TOO_SMALL" when nothing fits after buffer.
  - Deterministic ordering maintained.
- Versioning and public API
  - Bumped to 0.1.1 (pyproject.toml and src/virtual_context_mcp/__init__.py).
  - Slimmed __init__ public API to avoid importing heavy optional modules.
- Dependencies minimized
  - Core deps only in pyproject.toml (mcp, tiktoken, pydantic, pyyaml).
  - Heavy deps moved to extras ([vectors], [qdrant], [neo4j], [spacy]).
  - requirements.txt trimmed to core + test deps.
- Cline configuration checked in
  - .vscode/cline.json configured to launch stdio server with env {CONTEXT_MAX_TOKENS: "12000", PYTHONUNBUFFERED: "1"}.
- Docker alignment
  - docker/Dockerfile sets PYTHONUNBUFFERED=1, removes spaCy model download, keeps stdio CMD.
- Documentation updated
  - README reflects envelope format, version 0.1.1, Cline/Claude Desktop snippets, demo task flow, truncation/budget policy.

Files modified/added
- src/virtual_context_mcp/server.py: standardized envelopes, meta, pack_stats, BUDGET_TOO_SMALL.
- src/virtual_context_mcp/__init__.py: version to 0.1.1; simplified exports.
- pyproject.toml: version to 0.1.1; dependencies minimized; optional extras added.
- requirements.txt: trimmed to minimal core + test dependencies.
- .vscode/cline.json: added with cont3xt server config for Cline.
- docker/Dockerfile: unbuffered logs; removed heavy model download; stdio CMD retained.
- README.md: envelopes, usage, demo task, policies.

How to use (local)
- pip install -e .
- virtual-context-mcp --init-db
- Launch Cline; the “cont3xt” MCP server will be available with tools: memory_upsert, search_memory, context_pack, health_ping.
This completes M1 items required for Cline integration: stable stdio server, four tools, repo config for Cline checked in, and deterministic/non-flaky token budgeting with clear meta.