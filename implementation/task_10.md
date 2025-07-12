### TASK 10: Memory Retrieval and Relevance Scoring

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

**Objective**: Implement intelligent memory retrieval combining vector and graph search  
**Files to Create**: `src/virtual_context_mcp/retrieval/relevance_scorer.py`  
**Dependencies**: Task 9 complete

**Specifications**:

```python
# Required memory retrieval interface
from typing import List, Dict, Any, Tuple, Optional
from ..chunking.chunk import ContextChunk, MemoryEntry
from ..memory.vector_store import VectorStore
from ..memory.graph_store import GraphStore

class RelevanceScorer:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        pass
    
    async def score_memory_relevance(self, 
                                   memory_chunk: ContextChunk, 
                                   current_context: str,
                                   session_id: str) -> float:
        """Calculate relevance score for memory chunk"""
        # Combine multiple scoring factors:
        # 1. Semantic similarity (40% weight)
        # 2. Entity overlap (30% weight)  
        # 3. Temporal relevance (20% weight)
        # 4. Graph connectivity (10% weight)
        # Return score 0.0 to 1.0
        pass
    
    async def get_relevant_memories(self, 
                                  current_input: str,
                                  session_id: str,
                                  max_memories: int = 8,
                                  min_score: float = 0.3) -> List[MemoryEntry]:
        """Get most relevant memories for current input"""
        # 1. Vector search for semantic similarity
        # 2. Extract entities from current input
        # 3. Graph search for connected entities
        # 4. Score and rank all candidate memories
        # 5. Return top memories above min_score
        pass
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        pass
    
    async def calculate_entity_overlap(self, chunk: ContextChunk, current_text: str) -> float:
        """Calculate entity overlap score"""
        # Extract entities from both texts
        # Calculate Jaccard similarity of entity sets
        pass
    
    def calculate_temporal_relevance(self, chunk: ContextChunk) -> float:
        """Calculate temporal relevance (recent = higher score)"""
        # Exponential decay based on chunk age
        pass
    
    async def calculate_graph_connectivity(self, chunk: ContextChunk, current_entities: List[str], session_id: str) -> float:
        """Calculate graph connectivity score"""
        # How connected are chunk entities to current entities?
        pass
```

**Scoring Algorithm Specifications**:

```python
# Required scoring weights and calculations
RELEVANCE_WEIGHTS = {
    "semantic": 0.4,
    "entity_overlap": 0.3,
    "temporal": 0.2,
    "graph_connectivity": 0.1
}

# Temporal decay function
# score = exp(-age_hours / 24)  # 24-hour half-life

# Entity overlap calculation
# jaccard_similarity = len(intersection) / len(union)

# Graph connectivity
# connectivity = sum(shared_relationships) / total_possible_relationships
```

**Acceptance Criteria**:

- Relevance scores always between 0.0 and 1.0
- Semantic similarity component works correctly
- Entity overlap calculation accurate
- Temporal scoring favors recent memories appropriately
- Graph connectivity identifies related story elements
- Combined scoring produces intuitive rankings
- Performance under 200ms for typical queries