### TASK 9: Neo4j Knowledge Graph Integration

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

**Objective**: Store and query story relationships in Neo4j
**Files to Create**: `src/virtual_context_mcp/memory/graph_store.py`
**Dependencies**: Task 8 complete

**Specifications**:
```python
# Required Neo4j integration interface
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional
from ..entities.story_entities import StoryEntity

class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        pass
    
    async def init_schema(self) -> None:
        """Create constraints and indexes"""
        # Character name uniqueness
        # Location name uniqueness
        # Plot thread ID uniqueness
        pass
    
    async def store_entities(self, entities: Dict[str, List[StoryEntity]], chunk_id: str, session_id: str) -> None:
        """Store entities and relationships"""
        # Create nodes for new entities
        # Create relationships between entities
        # Link entities to chunk/session
        pass
    
    async def create_character(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update character node"""
        pass
    
    async def create_location(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update location node"""
        pass
    
    async def create_relationship(self, entity1: str, relationship: str, entity2: str, session_id: str) -> None:
        """Create relationship between entities"""
        # Examples: (Character)-[:KNOWS]->(Character)
        #          (Character)-[:LOCATED_IN]->(Location)
        pass
    
    async def get_character_relationships(self, character_name: str, session_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a character"""
        pass
    
    async def get_connected_entities(self, entity_name: str, session_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities connected within depth levels"""
        pass
    
    async def search_by_relationship(self, relationship_type: str, session_id: str) -> List[Dict[str, Any]]:
        """Find entities by relationship type"""
        pass
```

**Required Cypher Queries**:
```cypher
-- Schema creation
CREATE CONSTRAINT character_name_session FOR (c:Character) REQUIRE (c.name, c.session_id) IS UNIQUE;
CREATE CONSTRAINT location_name_session FOR (l:Location) REQUIRE (l.name, l.session_id) IS UNIQUE;

-- Entity creation
MERGE (c:Character {name: $name, session_id: $session_id})
SET c += $attributes
RETURN c;

-- Relationship creation
MATCH (a {name: $entity1, session_id: $session_id})
MATCH (b {name: $entity2, session_id: $session_id})
MERGE (a)-[r:$relationship]->(b)
RETURN r;

-- Connected entities query
MATCH (start {name: $entity_name, session_id: $session_id})
MATCH (start)-[*1..$depth]-(connected)
RETURN DISTINCT connected.name, labels(connected), connected;
```

**Acceptance Criteria**:
- Database schema creates successfully
- Entity creation handles duplicates properly
- Relationship creation works between all entity types
- Session isolation maintained
- Graph traversal queries return correct results
- Performance acceptable for real-time queries