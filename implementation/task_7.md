### TASK 7: Vector Store Integration (Qdrant)

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

**Objective**: Implement semantic similarity search using Qdrant
**Files to Create**: `src/virtual_context_mcp/memory/vector_store.py`
**Dependencies**: Task 6 complete

**Specifications**:
```python
# Required Qdrant integration interface
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from ..chunking.chunk import ContextChunk

class VectorStore:
    def __init__(self, 
                 qdrant_url: str,
                 collection_name: str = "story_chunks",
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Qdrant client and embedding model"""
        pass
    
    async def init_collection(self) -> None:
        """Create collection if it doesn't exist"""
        # Vector size should match sentence transformer model
        # Use Cosine distance
        pass
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    async def store_chunk(self, chunk: ContextChunk) -> None:
        """Store chunk with embedding in Qdrant"""
        # Generate embedding if not present
        # Store with metadata (session_id, timestamp, chunk_type)
        pass
    
    async def search_similar(self, 
                           query_text: str, 
                           session_id: Optional[str] = None,
                           limit: int = 5,
                           score_threshold: float = 0.7) -> List[Tuple[ContextChunk, float]]:
        """Search for similar chunks"""
        # Return chunks with similarity scores
        # Filter by session_id if provided
        # Only return results above score_threshold
        pass
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from vector store"""
        pass
```

**Required Qdrant Configuration**:
```python
# Collection configuration
VECTOR_CONFIG = VectorParams(
    size=384,  # all-MiniLM-L6-v2 embedding size
    distance=Distance.COSINE
)

# Point structure for storage
# {
#     "id": chunk_id,
#     "vector": embedding_vector,
#     "payload": {
#         "session_id": str,
#         "content": str,
#         "timestamp": str,
#         "chunk_type": str,
#         "token_count": int
#     }
# }
```

**Acceptance Criteria**:
- Qdrant collection initializes correctly
- Embeddings generate consistently
- Similarity search returns relevant results
- Session filtering works properly
- Score thresholds filter correctly
- Deletion operations work without errors