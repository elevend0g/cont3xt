 ### TASK 4: Token Counting and Basic Chunking

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

**Objective**: Implement accurate token counting and basic chunking functionality
**Files to Create**: `src/virtual_context_mcp/chunking/tokenizer.py`
**Dependencies**: Task 3 complete

**Specifications**:
```python
# Required interface for tokenizer
import tiktoken
from typing import List, Tuple

class TokenCounter:
    def __init__(self, model_name: str = "cl100k_base"):
        """Initialize with specific tokenizer model"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Return exact token count for text"""
        pass
    
    def split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks of max_tokens size"""
        pass
    
    def find_split_boundary(self, text: str, target_tokens: int) -> int:
        """Find best character position to split at target_tokens"""
        # Prefer sentence boundaries, then word boundaries
        pass

class BasicChunker:
    def __init__(self, token_counter: TokenCounter, chunk_size: int = 3200):
        pass
    
    def chunk_text(self, text: str, preserve_boundaries: bool = True) -> List[str]:
        """Split text into semantic chunks"""
        # Must preserve sentence boundaries when possible
        # Must not exceed chunk_size tokens per chunk
        pass
    
    def extract_oldest_chunks(self, chunks: List[str], target_tokens: int) -> List[str]:
        """Extract oldest chunks totaling approximately target_tokens"""
        pass
```

**Acceptance Criteria**:
- Token counts match OpenAI/Claude tokenizers exactly
- Chunks never exceed specified token limits
- Sentence boundaries preserved when possible
- Text reconstruction perfect (no lost characters)