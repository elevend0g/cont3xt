### TASK 11: Context Manager Core

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


**Objective**: Implement main context management orchestration  
**Files to Create**: `src/virtual_context_mcp/context_manager.py`  
**Dependencies**: Task 10 complete

**Specifications**:

```python
# Required context manager interface
from typing import Dict, List, Optional, Tuple
from .chunking.chunk import ContextChunk, ContextWindow
from .pressure_valve import PressureReliefValve
from .memory.sqlite_store import SQLiteStore
from .memory.vector_store import VectorStore
from .memory.graph_store import GraphStore
from .retrieval.relevance_scorer import RelevanceScorer
from .config.settings import Config

class ContextManager:
    def __init__(self, config: Config):
        """Initialize all components"""
        # Initialize all storage and processing components
        pass
    
    async def initialize(self) -> None:
        """Initialize databases and connections"""
        pass
    
    async def process_interaction(self, 
                                user_input: str, 
                                assistant_response: str,
                                session_id: str) -> ContextWindow:
        """Process a complete user-assistant interaction"""
        # 1. Create chunks from interaction
        # 2. Add to current context window
        # 3. Check pressure and trigger relief if needed
        # 4. Store interaction in memory systems
        # 5. Return updated context window
        pass
    
    async def build_context_window(self, 
                                 current_input: str,
                                 session_id: str,
                                 max_tokens: int) -> ContextWindow:
        """Build optimal context window for LLM inference"""
        # 1. Get recent conversation history
        # 2. Retrieve relevant memories
        # 3. Assemble context window under token limit
        # 4. Prioritize by relevance and recency
        pass
    
    async def add_interaction(self, 
                            user_input: str, 
                            assistant_response: str,
                            session_id: str) -> None:
        """Add new interaction to context"""
        pass
    
    async def get_current_pressure(self, session_id: str) -> float:
        """Get current context pressure for session"""
        pass
    
    async def force_relief(self, session_id: str) -> int:
        """Force context relief and return tokens removed"""
        pass
    
    async def get_context_stats(self, session_id: str) -> Dict[str, Any]:
        """Get context statistics for session"""
        # Return: current_tokens, pressure, total_chunks, last_relief, etc.
        pass
```

**Context Assembly Algorithm**:

```python
# Required context assembly strategy
async def build_context_window(self, current_input: str, session_id: str, max_tokens: int) -> ContextWindow:
    # 1. Reserve tokens for current input (actual + buffer)
    available_tokens = max_tokens - self.estimate_input_tokens(current_input) - 500
    
    # 2. Get recent conversation (70% of available tokens)
    recent_tokens = int(available_tokens * 0.7)
    recent_chunks = await self.get_recent_chunks(session_id, recent_tokens)
    
    # 3. Get relevant memories (30% of available tokens)
    memory_tokens = available_tokens - sum(chunk.token_count for chunk in recent_chunks)
    relevant_memories = await self.relevance_scorer.get_relevant_memories(
        current_input, session_id, max_tokens=memory_tokens
    )
    
    # 4. Combine and create context window
    all_chunks = recent_chunks + [entry.chunk for entry in relevant_memories]
    return ContextWindow(chunks=all_chunks, session_id=session_id)
```

**Acceptance Criteria**:

- Context windows never exceed max_tokens
- Pressure relief triggers at correct thresholds
- Memory retrieval enhances context appropriately
- Recent conversation always prioritized
- Context assembly completes under 100ms
- All database operations properly async
- Error handling for all failure modes