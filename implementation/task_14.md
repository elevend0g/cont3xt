### TASK 14: Integration Testing Suite

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

**Objective**: Create comprehensive integration tests  
**Files to Create**: `tests/integration/test_full_story_workflow.py`  
**Dependencies**: Task 13 complete

**Specifications**:

```python
# Required integration test scenarios
import pytest
import asyncio
from src.virtual_context_mcp.context_manager import ContextManager
from src.virtual_context_mcp.config.settings import Config

class TestStoryWorkflow:
    """Test complete story writing workflow"""
    
    @pytest.fixture
    async def context_manager(self):
        """Setup test context manager"""
        # Use test configuration with in-memory databases
        pass
    
    @pytest.mark.asyncio
    async def test_pressure_relief_cycle(self, context_manager):
        """Test that pressure relief works correctly"""
        # 1. Fill context to 80% capacity
        # 2. Verify pressure relief triggers
        # 3. Verify context reduced to ~60% capacity
        # 4. Verify no data loss
        # 5. Verify memories stored correctly
        pass
    
    @pytest.mark.asyncio
    async def test_character_consistency(self, context_manager):
        """Test character memory across context relief"""
        # 1. Introduce character with specific traits
        # 2. Fill context and trigger relief
        # 3. Reference character again
        # 4. Verify character memories retrieved
        # 5. Verify trait consistency maintained
        pass
    
    @pytest.mark.asyncio
    async def test_plot_thread_tracking(self, context_manager):
        """Test plot thread continuity"""
        # 1. Establish multiple plot threads
        # 2. Advance plots across multiple relief cycles
        # 3. Verify plot connections maintained
        # 4. Verify plot resolution tracking
        pass
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_accuracy(self, context_manager):
        """Test memory retrieval relevance"""
        # 1. Create story with specific details
        # 2. Trigger multiple relief cycles
        # 3. Query for specific story elements
        # 4. Verify correct memories retrieved
        # 5. Verify relevance scores appropriate
        pass
    
    @pytest.mark.asyncio 
    async def test_long_conversation_performance(self, context_manager):
        """Test performance with 500+ interactions"""
        # 1. Simulate 500 story interactions
        # 2. Measure response times throughout
        # 3. Verify memory usage doesn't grow unbounded
        # 4. Verify retrieval performance remains constant
        pass
```

**Performance Benchmarks**:

```python
# Required performance targets for tests
PERFORMANCE_TARGETS = {
    "context_assembly": 100,  # ms
    "pressure_relief": 500,   # ms
    "memory_search": 200,     # ms
    "entity_extraction": 50,  # ms per chunk
    "graph_query": 100,       # ms
    "vector_search": 150,     # ms
}

# Memory usage targets
MEMORY_TARGETS = {
    "max_context_size": 12000,     # tokens
    "relief_threshold": 9600,      # tokens (80%)
    "post_relief_size": 7200,      # tokens (60%)
    "chunks_per_relief": 2,        # number of chunks removed
}
```

**Acceptance Criteria**:

- All integration tests pass consistently
- Performance targets met under test conditions
- Memory usage stays within bounds
- Error conditions handled gracefully
- Test coverage >90% for core functionality
- Tests can run in CI/CD environment