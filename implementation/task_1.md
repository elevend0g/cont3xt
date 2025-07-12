### TASK 1: Project Foundation Setup

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

**Objective**: Create basic project structure with dependencies
**Files to Create**: `pyproject.toml`, `requirements.txt`, `src/virtual_context_mcp/__init__.py`
**Dependencies**: None

**Specifications**:
```toml
# pyproject.toml requirements
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "virtual-context-mcp"
version = "0.1.0"
description = "Virtual infinite context MCP server for AI agents"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0.0",
    "asyncio",
    "tiktoken>=0.5.0",
    "sentence-transformers>=2.2.0",
    "qdrant-client>=1.7.0",
    "neo4j>=5.15.0",
    "sqlite3",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "spacy>=3.7.0"
]
```

**Acceptance Criteria**:
- Valid Python package structure
- All dependencies properly specified
- Package installable with `pip install -e .`
