### TASK 12: MCP Server Implementation

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


**Objective**: Implement MCP protocol server with story-specific tools  
**Files to Create**: `src/virtual_context_mcp/server.py`  
**Dependencies**: Task 11 complete

**Specifications**:

```python
# Required MCP server interface
import mcp
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from .context_manager import ContextManager
from .config.settings import Config, load_config

class VirtualContextMCPServer:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP server with context manager"""
        pass
    
    async def setup_server(self) -> Server:
        """Configure MCP server with tools and resources"""
        server = Server("virtual-context")
        
        # Register all tools and resources
        await self.register_tools(server)
        await self.register_resources(server)
        
        return server
```

**Required MCP Tools**:

```python
# Tool specifications for MCP registration

@server.call_tool()
async def continue_story(arguments: dict) -> List[TextContent]:
    """Add story content and manage context automatically"""
    # Required arguments: user_input, assistant_response, session_id
    # Returns: context_window_summary, pressure_status
    pass

@server.call_tool()  
async def search_story_memory(arguments: dict) -> List[TextContent]:
    """Search story memory for relevant details"""
    # Required arguments: query, session_id
    # Optional: entity_type, max_results
    # Returns: relevant_memories with scores
    pass

@server.call_tool()
async def get_character_info(arguments: dict) -> List[TextContent]:
    """Get information about story characters"""
    # Required arguments: character_name, session_id
    # Returns: character_details, relationships, recent_mentions
    pass

@server.call_tool()
async def get_plot_threads(arguments: dict) -> List[TextContent]:
    """Get active plot threads and their status"""
    # Required arguments: session_id
    # Returns: active_plots, resolved_plots, plot_connections
    pass

@server.call_tool()
async def get_context_stats(arguments: dict) -> List[TextContent]:
    """Get context management statistics"""
    # Required arguments: session_id
    # Returns: pressure, tokens, relief_history, memory_counts
    pass
```

**Required MCP Resources**:

```python
# Resource specifications for MCP registration

@server.list_resources()
async def list_resources() -> List[Resource]:
    return [
        Resource(
            uri="story://current-context",
            name="Current Story Context",
            description="Active context window for story writing",
            mimeType="application/json"
        ),
        Resource(
            uri="story://characters",
            name="Story Characters",
            description="All characters in current story session",
            mimeType="application/json"
        ),
        Resource(
            uri="story://memory-stats",
            name="Memory Statistics", 
            description="Context management and memory statistics",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read specific resource content"""
    # Handle each resource URI appropriately
    pass
```

**Acceptance Criteria**:

- MCP server starts without errors
- All tools respond to calls correctly
- All resources return valid JSON
- Error handling for invalid arguments
- Session management works properly
- Tool responses include proper status information
- Server can handle concurrent tool calls