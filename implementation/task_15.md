### TASK 15: CLI and Entry Point

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

**Objective**: Create command-line interface and package entry point  
**Files to Create**: `src/virtual_context_mcp/__main__.py`, `src/virtual_context_mcp/cli.py`  
**Dependencies**: Task 14 complete

**Specifications**:

```python
# Required CLI interface
import argparse
import asyncio
import logging
from pathlib import Path
from .server import VirtualContextMCPServer
from .config.settings import load_config

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Virtual Context MCP Server")
    parser.add_argument("--config", type=str, default="configs/novel_writing.yaml",
                       help="Configuration file path")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host to bind server")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind server")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--init-db", action="store_true",
                       help="Initialize databases and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Run appropriate command
    if args.init_db:
        asyncio.run(init_databases(args.config))
    else:
        asyncio.run(run_server(args))

async def run_server(args):
    """Run the MCP server"""
    # Load configuration
    # Initialize server
    # Start serving
    pass

async def init_databases(config_path: str):
    """Initialize all databases"""
    # Create database schemas
    # Setup initial data if needed
    pass

if __name__ == "__main__":
    main()
```

**Package Entry Point Configuration**:

```toml
# Add to pyproject.toml
[project.scripts]
virtual-context-mcp = "virtual_context_mcp.__main__:main"

[project.entry-points."mcp.servers"]
virtual-context = "virtual_context_mcp.server:VirtualContextMCPServer"
```

**Acceptance Criteria**:

- CLI runs without errors
- All command-line options work correctly
- Configuration loading works properly
- Database initialization completes successfully
- Server starts and accepts connections
- Debug logging provides useful information
- Help text is clear and accurate