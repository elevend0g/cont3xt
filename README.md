# cont3xt

Virtual Infinite Context for Agents and LLMs  
An MCP server that maintains a continuous rolling context window, always surfacing the most relevant memories for the task at hand — while respecting your token budget.

---

## Current Status

This repository includes an MVP MCP stdio server suitable for integration with the Cline extension for VS Code. It exposes a minimal, SQLite-only toolset over stdio with zero external services required by default.

- Version: 0.1.1
- Transport: MCP stdio
- Default storage: SQLite at ./data/memory.db (configurable; example config uses ./data/novel_memory.db)
- Optional backends: Qdrant / Neo4j (disabled by default in this MVP)
- Tools implemented:
  - memory_upsert
  - search_memory
  - context_pack
  - health_ping

Notes:
- Tool names use underscores for compatibility with the Python server decorator. Dotted aliases (e.g., context.pack) can be added later.
- All tools return a standard JSON envelope as text content:
  {
    "ok": boolean,
    "data": {...},
    "meta": { "duration_ms": number, "backend": "sqlite", "version": "0.1.1" }
  }

---

## Quickstart

### 1) Install

Requires Python 3.11+

```bash
git clone https://github.com/elevend0g/cont3xt.git
cd cont3xt
pip install -r requirements.txt
pip install -e .
```

### 2) Initialize local storage (SQLite)

```bash
virtual-context-mcp --init-db
```

This creates ./data/memory.db if it doesn’t exist (or the path specified by config/env).

### 3) Run (stdio transport)

```bash
virtual-context-mcp
```

This starts the MCP server over stdio.

---

## Using with MCP Clients

### Cline (VS Code)

Add an MCP server named cont3xt that launches the stdio server:

Example configuration shape (adapt for your Cline settings UI/JSON):

```json
{
  "mcpServers": {
    "cont3xt": {
      "command": "virtual-context-mcp",
      "args": ["--config", "configs/novel_writing.yaml"],
      "env": {
        "CONTEXT_MAX_TOKENS": "12000",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Notes:
- Ensure virtual-context-mcp is in PATH (pip install -e . creates the console script).
- The --config argument is optional; environment variables can override values (see Configuration).

### Claude Desktop

```json
{
  "mcpServers": {
    "cont3xt": {
      "command": "virtual-context-mcp",
      "args": ["--config", "configs/novel_writing.yaml"],
      "env": {
        "CONTEXT_MAX_TOKENS": "12000",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## Tools (MVP)

All tools return a JSON envelope as text content.

Envelope:
```json
{
  "ok": true,
  "data": { /* tool-specific payload */ },
  "meta": {
    "duration_ms": 5.23,
    "backend": "sqlite",
    "version": "0.1.1"
  }
}
```

### 1) memory_upsert

Upsert memory for a session from either combined content or a user/assistant pair.

Arguments:
- session_id: string (required)
- content: string (optional; direct content)
- user_input: string (optional; used if content not provided)
- assistant_response: string (optional; used if content not provided)

Response (example):
```json
{
  "ok": true,
  "data": {
    "session_id": "sess-1",
    "ids": ["b1c...f"],
    "count": 1
  },
  "meta": {
    "duration_ms": 1.23,
    "backend": "sqlite",
    "version": "0.1.1"
  }
}
```

### 2) search_memory

Simple substring search with a recency-biased score over recent SQLite chunks.

Arguments:
- session_id: string (required)
- query: string (required)
- max_results: number (optional, default 10)

Response (example):
```json
{
  "ok": true,
  "data": {
    "query": "emerald eyes",
    "results_count": 2,
    "results": [
      {
        "chunk_id": "b1c...f",
        "score": 1.7,
        "timestamp": "2025-08-09T17:30:00.000000",
        "token_count": 142,
        "preview": "User: ... Assistant: ..."
      }
    ]
  },
  "meta": {
    "duration_ms": 2.45,
    "backend": "sqlite",
    "version": "0.1.1"
  }
}
```

### 3) context_pack

Packs a token-budgeted context using recent conversation chunks. Reserves a small buffer and includes the current input if provided.

Arguments:
- session_id: string (required)
- current_input: string (optional, default "")
- budget_tokens: number (optional, defaults to CONTEXT_MAX_TOKENS)

Response (example):
```json
{
  "ok": true,
  "data": {
    "pack": {
      "schema_version": "ctx.v1",
      "budget_tokens": 12000,
      "used_tokens": 2834,
      "sections": [
        {"role": "user", "title": "Current Input", "content": "..."},
        {
          "role": "context",
          "title": "Conversation (2025-08-09T17:30:00.000000)",
          "content": "...",
          "chunk_id": "b1c...f",
          "token_count": 142
        }
      ],
      "provenance": {
        "retriever": "recent-only",
        "stores": {"sqlite": "./data/memory.db"}
      },
      "pack_stats": {
        "num_candidates": 12,
        "kept": 4,
        "dropped": 8,
        "buffer_tokens": 500
      }
    },
    "meta": {
      "input_tokens": 120,
      "output_tokens": 2834,
      "budget": 12000,
      "sources": ["b1c...f", "a9d...1"]
    }
  },
  "meta": {
    "duration_ms": 7.89,
    "backend": "sqlite",
    "version": "0.1.1"
  }
}
```

Notes:
- If nothing fits within the budget (after a fixed buffer of 500 tokens), the pack includes: `"reason": "BUDGET_TOO_SMALL"`.
- Truncation policy: respect budget strictly; sections are assembled most-recent-first, then presented chronologically for readability.

### 4) health_ping

Returns basic status and configuration fingerprint.

Arguments: none

Response (example):
```json
{
  "ok": true,
  "data": {
    "server": "cont3xt",
    "version": "0.1.1",
    "datetime": "2025-08-09T17:32:00.000000",
    "config": {
      "sqlite_path": "./data/memory.db",
      "max_tokens": 12000,
      "token_model": "cl100k_base",
      "optional_backends": {
        "qdrant": false,
        "neo4j": false
      }
    }
  },
  "meta": {
    "duration_ms": 0.41,
    "backend": "sqlite",
    "version": "0.1.1"
  }
}
```

---

## Cline Demo Task (Code Gen Flow)

1) In Cline, run a task like “Improve function X in file Y; add a unit test and make tests pass.”
2) Cline should:
   - Call `search_memory(session_id, query="project goals")`
   - Call `context_pack(session_id, current_input="<file diff request>", budget_tokens=12000)`
   - Generate plan → apply small diff → run tests
   - Call `memory_upsert` with `(user_input, assistant_response)` to persist the session outcome
3) Acceptance:
   - From a clean clone, `pip install -e .`, launch Cline, and complete a guided code edit PR in one go using the above flow.

---

## Configuration

Defaults are provided in code, optionally loaded from a YAML file (e.g., configs/novel_writing.yaml) and overridden by environment variables.

Environment variable overrides (examples):
- CONTEXT_MAX_TOKENS (default 12000)
- CONTEXT_PRESSURE_THRESHOLD (default 0.8)
- CONTEXT_RELIEF_PERCENTAGE (default 0.4)
- CONTEXT_CHUNK_SIZE (default 3200)
- CONTEXT_TOKEN_MODEL (default cl100k_base)

- DATABASE_SQLITE_PATH (default ./data/memory.db)
- DATABASE_QDRANT_URL
- DATABASE_QDRANT_COLLECTION
- DATABASE_NEO4J_URL
- DATABASE_NEO4J_USER
- DATABASE_NEO4J_PASSWORD

For the MVP, only SQLite is used. Qdrant / Neo4j are intentionally not required; future versions can enable them via env/config flags.

Truncation policy:
- A fixed buffer of 500 tokens is reserved.
- Packing order: include current_input (if provided) and then pack recent conversation chunks until the budget is met.
- If no sections can be included, the pack includes `"reason": "BUDGET_TOO_SMALL"`.

---

## Development

Entry points:
- Console script: virtual-context-mcp → virtual_context_mcp.__main__:main
- MCP server class: virtual_context_mcp.server:VirtualContextMCPServer

Local logs: standard output (set PYTHONUNBUFFERED=1 for real-time logs). SQLite file: ./data/memory.db by default.

Run unit/integration tests as needed (note that legacy story-focused modules may not reflect the MVP surface area).

---

## Security

- MVP runs in a safe default mode: only persists to local SQLite.
- No filesystem ingestion or external network writes by default.

---

## Roadmap

- Optional Qdrant/Neo4j integration behind env flags with graceful fallback.
- Aliased/dotted tool names for compatibility (e.g., context.pack).
- Richer retrieval (hybrid, deduplication), entity/graph memory, and pressure relief policies.
- DevContainer / Docker image for reproducible environments.

---

## License

MIT © elevend0g
