### TASK 6: Pressure Relief Valve Implementation

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


**Objective**: Implement core context pressure relief mechanism
**Files to Create**: `src/virtual_context_mcp/pressure_valve.py`
**Dependencies**: Task 5 complete

**Specifications**:
```python
# Required pressure valve interface
from typing import List, Tuple
from .chunking.chunk import ContextChunk, ContextWindow
from .chunking.tokenizer import TokenCounter, BasicChunker
from .memory.sqlite_store import SQLiteStore

class PressureReliefValve:
    def __init__(self, 
                 token_counter: TokenCounter,
                 chunker: BasicChunker,
                 storage: SQLiteStore,
                 pressure_threshold: float = 0.8,
                 relief_percentage: float = 0.4):
        pass
    
    def calculate_pressure(self, context_window: ContextWindow, max_tokens: int) -> float:
        """Calculate pressure ratio (0.0 to 1.0)"""
        # Return: current_tokens / max_tokens
        pass
    
    async def needs_relief(self, context_window: ContextWindow, max_tokens: int) -> bool:
        """Check if pressure relief is needed"""
        pass
    
    async def execute_relief(self, context_window: ContextWindow, max_tokens: int) -> ContextWindow:
        """Execute pressure relief and return trimmed context"""
        # 1. Calculate tokens to remove (max_tokens * relief_percentage)
        # 2. Select oldest chunks totaling that amount
        # 3. Store selected chunks in SQLite
        # 4. Remove chunks from context window
        # 5. Return updated context window
        pass
    
    def select_chunks_for_relief(self, context_window: ContextWindow, target_tokens: int) -> List[ContextChunk]:
        """Select optimal chunks for removal"""
        # Prefer oldest chunks
        # Try to select complete semantic units
        # Must total approximately target_tokens
        pass
```

**Acceptance Criteria**:
- Pressure calculation is accurate
- Relief triggers only at correct thresholds
- Chunk selection preserves semantic coherence
- Storage operations complete successfully
- Context window size reduced to target level
- No data loss during relief operations