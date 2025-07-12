### TASK 3: Core Data Models

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

**Objective**: Define core data structures for chunks and memory
**Files to Create**: `src/virtual_context_mcp/chunking/chunk.py`
**Dependencies**: Task 2 complete

**Specifications**:
```python
# Required data models with exact field specifications
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class ContextChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: str
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.now)
    chunk_type: str = "conversation"  # conversation, character, plot, setting
    entities: Optional[Dict[str, List[str]]] = None
    embedding: Optional[List[float]] = None
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize for storage"""
        pass
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ContextChunk":
        """Deserialize from storage"""
        pass

class MemoryEntry(BaseModel):
    chunk_id: str
    relevance_score: float
    retrieval_reason: str  # "semantic", "graph", "temporal"
    
class ContextWindow(BaseModel):
    chunks: List[ContextChunk]
    total_tokens: int
    session_id: str
    
    def calculate_tokens(self) -> int:
        """Calculate total token count"""
        pass
    
    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add chunk and update token count"""
        pass
    
    def remove_chunks(self, chunk_ids: List[str]) -> List[ContextChunk]:
        """Remove chunks and return them"""
        pass
```

**Acceptance Criteria**:
- All models validate with Pydantic
- Serialization/deserialization works correctly
- Token counting methods implemented
- Chunk manipulation methods work properly
