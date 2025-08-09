Completed a focused code review and implemented an MCP stdio MVP suitable for integration into the Cline extension for VS Code. Key outcomes and changes:

Summary of issues discovered during review:
- README and module drift: README referenced python -m cont3xt stdio and dotted tool names (context.pack, search.memory, memory.upsert, health.ping), but the codebase shipped virtual_context_mcp with story-centric tools instead.
- No stdio transport: __main__.py had a dummy loop and never started an MCP stdio transport.
- API mismatches: server.py referenced methods/fields that don’t exist or whose signatures had diverged (e.g., build_context_window signature, chunk_id vs id, retrieval_system usage).
- Heavy backend requirements: Qdrant/Neo4j/spaCy required in practice, causing failures or network/model downloads that are not ideal for an initial Cline integration.
- Data model gaps: BasicChunker returns List[str] but higher layers treated them as complex chunks in some places.

What was implemented (MVP for Cline):
1) Minimal stdio MCP server with a clean tool surface (SQLite-only)
   - Implemented in src/virtual_context_mcp/server.py
   - Exposes four tools aligned with the requested MVP:
     - memory_upsert(session_id, content | user_input+assistant_response)
     - search_memory(session_id, query, max_results=10)
     - context_pack(session_id, current_input="", budget_tokens=config)
     - health_ping()
   - Tools return JSON envelopes as TextContent for compatibility.

2) Proper stdio transport and run-loop
   - src/virtual_context_mcp/__main__.py now runs the server over stdio using mcp.server.stdio.stdio_server and server.run(read, write).
   - Works with the existing console script entry point (virtual-context-mcp).

3) Default to SQLite only (no external services required)
   - Uses SQLiteStore at config.database.sqlite_path for persistence.
   - Token counts from TokenCounter (tiktoken).
   - Qdrant/Neo4j are not initialized in this MVP; future toggles can be added later behind env flags.

4) Updated documentation for Cline
   - README.md now documents the stdio run mode, the four MVP tools, arguments and sample responses, and a Cline configuration snippet using the console script virtual-context-mcp.

5) Docker runtime alignment
   - docker/Dockerfile command now starts stdio mode via console script: CMD ["virtual-context-mcp", "--config", "configs/novel_writing.yaml"].

Files changed:
- src/virtual_context_mcp/server.py
  - Replaced previous story-specific server with cont3xt MVP tools (memory_upsert, search_memory, context_pack, health_ping), named server "cont3xt" for client visibility. Uses SQLite + TokenCounter only.
- src/virtual_context_mcp/__main__.py
  - Added proper stdio transport run-loop (stdio_server + server.run).
  - Provided --init-db path to initialize SQLite storage without running a server.
- README.md
  - Corrected installation, stdio usage, tool docs, and configuration for Cline and Claude Desktop.
- docker/Dockerfile
  - Updated CMD to run virtual-context-mcp for stdio transport.

Notes on compatibility:
- pyproject.toml already defines:
  - [project.scripts] virtual-context-mcp = "virtual_context_mcp.__main__:main"
  - [project.entry-points."mcp.servers"] virtual-context = "virtual_context_mcp.server:VirtualContextMCPServer"
- The server class now provides the expected MCP tool handlers and a stdio run loop via __main__.
- By default, SQLite path respects YAML config and env overrides. The sample configs/novel_writing.yaml points to ./data/novel_memory.db.

How to use with Cline:
- Ensure dependencies are installed (pip install -r requirements.txt && pip install -e .).
- Initialize SQLite (optional): virtual-context-mcp --init-db
- Configure Cline to launch:
  {
    "mcpServers": {
      "cont3xt": {
        "command": "virtual-context-mcp",
        "args": ["--config", "configs/novel_writing.yaml"],
        "env": { "CONTEXT_MAX_TOKENS": "12000" }
      }
    }
  }
- Tools available in Cline: memory_upsert, search_memory, context_pack, health_ping.

Limitations and follow-up recommendations:
- Tool naming: current underscore names are conventional with Python handlers; dotted aliases (context.pack) can be added later if desired.
- Retrieval: search_memory is substring + recency biased for MVP; hybrid retrieval (Qdrant, BM25, graph) can be reintroduced behind feature flags.
- Dependencies: consider moving heavy optional deps (spacy, qdrant, neo4j) to extras to reduce baseline install size.
- Tests: existing legacy tests use story-focused flows and likely won’t reflect MVP surface; add new unit tests for the four tools and a smoke test for stdio run.
- Config gating: introduce env flags (e.g., CTX_ENABLE_QDRANT, CTX_ENABLE_NEO4J) with graceful fallback.

This state makes cont3xt ready to integrate with the Cline extension as an MCP stdio server with a stable, zero-external-dependency MVP toolset, while preserving a path to richer retrieval and memory backends in future iterations.