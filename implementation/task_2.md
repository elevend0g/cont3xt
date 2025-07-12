### TASK 2: Configuration System

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

**Objective**: Create type-safe configuration management
**Files to Create**: `src/virtual_context_mcp/config/settings.py`, `configs/novel_writing.yaml`
**Dependencies**: Task 1 complete

**Specifications**:
```python
# Required configuration structure using Pydantic
from pydantic import BaseModel
from typing import Optional

class ContextConfig(BaseModel):
    max_tokens: int = 12000
    pressure_threshold: float = 0.8
    relief_percentage: float = 0.4
    chunk_size: int = 3200

class DatabaseConfig(BaseModel):
    sqlite_path: str = "./data/memory.db"
    qdrant_url: str = "http://localhost:6333"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

class Config(BaseModel):
    context: ContextConfig = ContextConfig()
    database: DatabaseConfig = DatabaseConfig()
    model_name: str = "all-MiniLM-L6-v2"
    debug: bool = False
```

**YAML Configuration Template**:
```yaml
# configs/novel_writing.yaml
context:
  max_tokens: 12000
  pressure_threshold: 0.8
  relief_percentage: 0.4
  chunk_size: 3200

database:
  sqlite_path: "./data/novel_memory.db"
  qdrant_url: "http://localhost:6333"
  neo4j_url: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "password"

model_name: "all-MiniLM-L6-v2"
debug: true
```

**Acceptance Criteria**:
- Configuration loads from YAML file
- Pydantic validation catches invalid values
- Environment variable override support
- Default values work without config file