"""MCP stdio server for cont3xt MVP (SQLite-only, minimal tools).

Implements four MCP tools suitable for Cline integration:
- context_pack
- search_memory
- memory_upsert
- health_ping

Notes:
- Stdio transport is started from __main__.py (async with stdio_server()).
- Qdrant/Neo4j are intentionally not required/initialized for MVP.
- Uses existing SQLiteStore and TokenCounter for persistence and token counts.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from mcp.server import Server
from mcp.types import TextContent, CallToolResult

from .config.settings import load_config, Config
from .memory.sqlite_store import SQLiteStore
from .chunking.tokenizer import TokenCounter
from .chunking.chunk import ContextChunk

logger = logging.getLogger(__name__)

# Standardized envelope builder for tool responses
from time import perf_counter
from . import __version__ as PACKAGE_VERSION

def _build_envelope(ok: bool, data: Dict[str, Any], duration_ms: float, backend: str = "sqlite", version: str = PACKAGE_VERSION, error: Optional[Dict[str, Any]] = None) -> CallToolResult:
    meta: Dict[str, Any] = {
        "duration_ms": round(duration_ms, 2),
        "backend": backend,
        "version": version,
    }
    if error:
        meta["error"] = error
    return CallToolResult([TextContent(type="text", text=json.dumps({"ok": ok, "data": data, "meta": meta}, indent=2))])


class _MemoryService:
    """Lightweight storage and packing helpers for MVP."""

    def __init__(self, config: Config):
        self.config = config
        self.sqlite = SQLiteStore(config.database.sqlite_path)
        self.tokens = TokenCounter(model_name=config.context.token_model)

    async def initialize(self) -> None:
        await self.sqlite.initialize()

    async def upsert_interaction(
        self,
        session_id: str,
        content: str,
        chunk_type: str = "conversation",
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        token_count = self.tokens.count_tokens(content)
        chunk = ContextChunk(
            session_id=session_id,
            content=content,
            token_count=token_count,
            chunk_type=chunk_type,
            entities=entities,
        )
        await self.sqlite.store_chunk(chunk)
        return chunk.id

    async def get_recent_chunks(self, session_id: str, limit: Optional[int] = None) -> List[ContextChunk]:
        return await self.sqlite.get_chunks_by_session(session_id, limit=limit)

    def pack_context(
        self,
        session_id: str,
        chunks_desc: List[ContextChunk],
        current_input: str,
        budget_tokens: int,
    ) -> Dict[str, Any]:
        """Build a simple budgeted pack: include recent conversation chunks until budget."""
        input_tokens = self.tokens.count_tokens(current_input or "")
        # Reserve a small buffer for system/meta tokens
        buffer_tokens = 500
        available = max(budget_tokens - input_tokens - buffer_tokens, 0)

        used = 0
        selected: List[ContextChunk] = []
        # chunks_desc is most-recent-first; pack from newest backwards
        for chunk in chunks_desc:
            if used + chunk.token_count > available:
                continue
            selected.append(chunk)
            used += chunk.token_count
            if used >= available:
                break

        # Present selected in chronological order for readability
        selected_sorted = sorted(selected, key=lambda c: c.timestamp)

        sections: List[Dict[str, Any]] = []
        if current_input:
            sections.append(
                {"role": "user", "title": "Current Input", "content": current_input}
            )

        if selected_sorted:
            # Collapse each chunk into a context section
            for ch in selected_sorted:
                preview = ch.content if len(ch.content) <= 2000 else (ch.content[:2000] + "...")
                sections.append(
                    {
                        "role": "context",
                        "title": f"Conversation ({ch.timestamp.isoformat()})",
                        "content": preview,
                        "chunk_id": ch.id,
                        "token_count": ch.token_count,
                    }
                )

        pack = {
            "schema_version": "ctx.v1",
            "budget_tokens": budget_tokens,
            "used_tokens": used + input_tokens,
            "sections": sections,
            "provenance": {
                "retriever": "recent-only",
                "stores": {"sqlite": self.config.database.sqlite_path},
            },
            "pack_stats": {
                "num_candidates": len(chunks_desc),
                "kept": len(selected_sorted),
                "dropped": max(len(chunks_desc) - len(selected_sorted), 0),
                "buffer_tokens": buffer_tokens,
            },
        }
        if available <= 0 and not current_input and len(selected_sorted) == 0:
            pack["reason"] = "BUDGET_TOO_SMALL"
        return pack

    @staticmethod
    def _score_match(text: str, query: str, recency_weight: float, ordinal: int) -> float:
        """Very basic scoring: occurrences + recency bias."""
        if not query:
            return 0.0
        lc = text.lower()
        q = query.lower()
        occurrences = lc.count(q)
        if occurrences == 0:
            return 0.0
        # Newer items (lower ordinal) score higher
        return occurrences + recency_weight / (1 + ordinal)


class VirtualContextMCPServer:
    """MCP Server exposing cont3xt MVP tools."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_config(config_path)
        self.server: Optional[Server] = None
        self.memory = _MemoryService(self.config)

    async def initialize(self) -> None:
        await self.memory.initialize()
        logger.info("cont3xt MVP server initialized (SQLite only)")

    async def setup_server(self) -> Server:
        if self.server is not None:
            return self.server

        await self.initialize()

        # Name shown to MCP clients
        server = Server("cont3xt")
        self.server = server

        # Tools
        self._register_memory_upsert()
        self._register_search_memory()
        self._register_context_pack()
        self._register_health_ping()

        logger.info("Registered cont3xt MVP tools: memory_upsert, search_memory, context_pack, health_ping")
        return server

    def _register_memory_upsert(self) -> None:
        @self.server.call_tool()
        async def memory_upsert(arguments: dict) -> CallToolResult:
            """
            Upsert memory for a session.
            Args:
              - session_id: str (required)
              - user_input: str (optional)
              - assistant_response: str (optional)
              - content: str (optional; direct content)
              - role: str (optional; if using 'content', e.g., 'system'|'user'|'assistant')
            """
            start = perf_counter()
            try:
                session_id = arguments.get("session_id")
                if not session_id:
                    raise ValueError("session_id is required")

                content = arguments.get("content")
                if content is None:
                    user_input = arguments.get("user_input", "")
                    assistant_response = arguments.get("assistant_response", "")
                    if not user_input and not assistant_response:
                        raise ValueError("Either content or user_input/assistant_response must be provided")
                    content = f"User: {user_input}\n\nAssistant: {assistant_response}".strip()
                # For MVP we ignore 'role' and store as conversation
                chunk_id = await self.memory.upsert_interaction(session_id=session_id, content=content)

                ms = (perf_counter() - start) * 1000.0
                data = {
                    "session_id": session_id,
                    "ids": [chunk_id],
                    "count": 1,
                }
                return _build_envelope(True, data, ms)

            except Exception as e:
                logger.exception("memory_upsert failed")
                ms = (perf_counter() - start) * 1000.0
                code = "BAD_ARGS" if isinstance(e, ValueError) else "INTERNAL_ERROR"
                err = {"code": code, "message": str(e)}
                return _build_envelope(False, {}, ms, error=err)

    def _register_search_memory(self) -> None:
        @self.server.call_tool()
        async def search_memory(arguments: dict) -> CallToolResult:
            """
            Search recent memory (SQLite).
            Args:
              - session_id: str (required)
              - query: str (required)
              - max_results: int (optional, default 10)
            """
            start = perf_counter()
            try:
                session_id = arguments.get("session_id")
                query = arguments.get("query")
                max_results = int(arguments.get("max_results", 10))
                if not session_id or not query:
                    raise ValueError("session_id and query are required")

                chunks_desc = await self.memory.get_recent_chunks(session_id)
                results: List[Tuple[ContextChunk, float]] = []
                for idx, ch in enumerate(chunks_desc):
                    score = self.memory._score_match(ch.content, query, recency_weight=1.0, ordinal=idx)
                    if score > 0:
                        results.append((ch, score))

                # Sort by score desc, take top N
                results.sort(key=lambda t: t[1], reverse=True)
                top = results[:max_results]

                ms = (perf_counter() - start) * 1000.0
                data = {
                    "query": query,
                    "results_count": len(top),
                    "results": [
                        {
                            "chunk_id": ch.id,
                            "score": score,
                            "timestamp": ch.timestamp.isoformat(),
                            "token_count": ch.token_count,
                            "preview": ch.content if len(ch.content) <= 500 else (ch.content[:500] + "..."),
                        }
                        for ch, score in top
                    ],
                }
                return _build_envelope(True, data, ms)

            except Exception as e:
                logger.exception("search_memory failed")
                ms = (perf_counter() - start) * 1000.0
                code = "BAD_ARGS" if isinstance(e, ValueError) else "INTERNAL_ERROR"
                err = {"code": code, "message": str(e)}
                return _build_envelope(False, {}, ms, error=err)

    def _register_context_pack(self) -> None:
        @self.server.call_tool()
        async def context_pack(arguments: dict) -> CallToolResult:
            """
            Build a token-budgeted context pack.
            Args:
              - session_id: str (required)
              - current_input: str (optional, default "")
              - budget_tokens: int (optional, default from config)
            """
            start = perf_counter()
            try:
                session_id = arguments.get("session_id")
                if not session_id:
                    raise ValueError("session_id is required")

                current_input = arguments.get("current_input", "") or ""
                budget_tokens = int(arguments.get("budget_tokens", self.config.context.max_tokens))

                chunks_desc = await self.memory.get_recent_chunks(session_id)
                pack = self.memory.pack_context(session_id, chunks_desc, current_input, budget_tokens)

                ms = (perf_counter() - start) * 1000.0
                sources = [s.get("chunk_id") for s in pack.get("sections", []) if s.get("role") == "context"]
                input_tokens = self.memory.tokens.count_tokens(current_input or "")
                data: Dict[str, Any] = {
                    "pack": pack,
                    "meta": {
                        "input_tokens": input_tokens,
                        "output_tokens": pack.get("used_tokens", 0),
                        "budget": budget_tokens,
                        "sources": sources,
                    },
                }
                if "reason" in pack:
                    data["reason"] = pack["reason"]
                return _build_envelope(True, data, ms)

            except Exception as e:
                logger.exception("context_pack failed")
                ms = (perf_counter() - start) * 1000.0
                code = "BAD_ARGS" if isinstance(e, ValueError) else "INTERNAL_ERROR"
                err = {"code": code, "message": str(e)}
                return _build_envelope(False, {}, ms, error=err)

    def _register_health_ping(self) -> None:
        @self.server.call_tool()
        async def health_ping(arguments: dict) -> CallToolResult:
            """
            Basic health and configuration fingerprint.
            """
            start = perf_counter()
            try:
                sqlite_path = self.config.database.sqlite_path
                ms = (perf_counter() - start) * 1000.0
                data = {
                    "server": "cont3xt",
                    "version": PACKAGE_VERSION,
                    "datetime": datetime.now().isoformat(),
                    "config": {
                        "sqlite_path": sqlite_path,
                        "max_tokens": self.config.context.max_tokens,
                        "token_model": self.config.context.token_model,
                        "optional_backends": {
                            "qdrant": False,
                            "neo4j": False,
                        },
                    },
                }
                return _build_envelope(True, data, ms)
            except Exception as e:
                logger.exception("health_ping failed")
                ms = (perf_counter() - start) * 1000.0
                code = "BAD_ARGS" if isinstance(e, ValueError) else "INTERNAL_ERROR"
                err = {"code": code, "message": str(e)}
                return _build_envelope(False, {}, ms, error=err)


# Convenience asynchronous entry for module execution if needed.
async def main() -> Server:
    app = VirtualContextMCPServer()
    server = await app.setup_server()
    logger.info("cont3xt MVP MCP server ready")
    return server
