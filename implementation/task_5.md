### TASK 5: SQLite Storage Implementation

## Project Structure Foundation

```
virtual-context-mcp/
├── src/
│   ├── virtual_context_mcp/
│   │   ├── __init__.py
│   │   ├── server.py              # MCP server entry point
│   │   ├── context_manager.py     # Core context management
│   │   ├── pressure_valve.py      # Context relief logic
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── store.py           # Memory storage interface
│   │   │   ├── vector_store.py    # Qdrant integration
│   │   │   ├── graph_store.py     # Neo4j integration
│   │   │   └── sqlite_store.py    # SQLite archival
│   │   ├── chunking/
│   │   │   ├── __init__.py
│   │   │   ├── tokenizer.py       # Token counting
│   │   │   ├── story_chunker.py   # Narrative-aware chunking
│   │   │   └── chunk.py           # Chunk data model
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── semantic_search.py # Vector similarity
│   │   │   ├── graph_search.py    # Knowledge graph queries
│   │   │   └── relevance_scorer.py # Memory relevance scoring
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py       # Entity extraction
│   │   │   └── story_entities.py  # Story-specific entities
│   │   └── config/
│   │       ├── __init__.py
│   │       └── settings.py        # Configuration management
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── configs/
│   └── novel_writing.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

**Objective**: Implement persistent storage for conversation chunks
**Files to Create**: `src/virtual_context_mcp/memory/sqlite_store.py`
**Dependencies**: Task 4 complete

**Specifications**:
```python
# Required SQLite schema and interface
import sqlite3
import asyncio
from typing import List, Optional
from ..chunking.chunk import ContextChunk

class SQLiteStore:
    def __init__(self, db_path: str):
        """Initialize SQLite database with schema"""
        pass
    
    async def init_database(self) -> None:
        """Create tables if they don't exist"""
        # Required tables:
        # chunks: id, session_id, content, token_count, timestamp, chunk_type, entities_json
        # sessions: session_id, created_at, last_active
        pass
    
    async def store_chunk(self, chunk: ContextChunk) -> None:
        """Store single chunk"""
        pass
    
    async def store_chunks(self, chunks: List[ContextChunk]) -> None:
        """Store multiple chunks in transaction"""
        pass
    
    async def get_chunks_by_session(self, session_id: str, limit: Optional[int] = None) -> List[ContextChunk]:
        """Retrieve chunks for session, most recent first"""
        pass
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[ContextChunk]:
        """Retrieve specific chunk"""
        pass
    
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks and return count deleted"""
        pass
```

**Required SQL Schema**:
```sql
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    chunk_type TEXT DEFAULT 'conversation',
    entities_json TEXT,
    embedding_json TEXT
);

CREATE INDEX idx_chunks_session ON chunks(session_id);
CREATE INDEX idx_chunks_timestamp ON chunks(timestamp);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Acceptance Criteria**:
- Database schema creates successfully
- All CRUD operations work correctly
- Transactions properly handle errors
- Async/await pattern implemented correctly
- Data integrity maintained across operations