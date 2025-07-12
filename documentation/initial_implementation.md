# Virtual Infinite Context MCP Server - LLM Agent Implementation Plan

## Agent-Optimized Development Strategy

This plan breaks down the implementation into discrete, atomic tasks suitable for LLM coding agents. Each task has clear specifications, acceptance criteria, and minimal interdependencies.

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

---

## Task Sequence for LLM Agent Implementation

### TASK 1: Project Foundation Setup
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

---

### TASK 2: Configuration System
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

---

### TASK 3: Core Data Models
**Objective**: Define core data structures for chunks and memory
**Files to Create**: `src/virtual_context_mcp/chunking/chunk.py`
**Dependencies**: Task 2 complete

**Specifications**:
```python
# Required data models with exact field specifications
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class ContextChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: str
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.now)
    chunk_type: str = "conversation"  # conversation, character, plot, setting
    entities: Optional[Dict[str, List[str]]] = None
    embedding: Optional[List[float]] = None
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize for storage"""
        pass
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ContextChunk":
        """Deserialize from storage"""
        pass

class MemoryEntry(BaseModel):
    chunk_id: str
    relevance_score: float
    retrieval_reason: str  # "semantic", "graph", "temporal"
    
class ContextWindow(BaseModel):
    chunks: List[ContextChunk]
    total_tokens: int
    session_id: str
    
    def calculate_tokens(self) -> int:
        """Calculate total token count"""
        pass
    
    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add chunk and update token count"""
        pass
    
    def remove_chunks(self, chunk_ids: List[str]) -> List[ContextChunk]:
        """Remove chunks and return them"""
        pass
```

**Acceptance Criteria**:
- All models validate with Pydantic
- Serialization/deserialization works correctly
- Token counting methods implemented
- Chunk manipulation methods work properly

---

### TASK 4: Token Counting and Basic Chunking
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

---

### TASK 5: SQLite Storage Implementation
**Objective**: Implement persistent storage for conversation chunks
**Files to Create**: `src/virtual_context_mcp/memory/sqlite_store.py`
**Dependencies**: Task 4 complete

**Specifications**:
```python
# Required SQLite schema and interface
import sqlite3
import asyncio
from typing import List, Optional
from ..chunking.chunk import ContextChunk

class SQLiteStore:
    def __init__(self, db_path: str):
        """Initialize SQLite database with schema"""
        pass
    
    async def init_database(self) -> None:
        """Create tables if they don't exist"""
        # Required tables:
        # chunks: id, session_id, content, token_count, timestamp, chunk_type, entities_json
        # sessions: session_id, created_at, last_active
        pass
    
    async def store_chunk(self, chunk: ContextChunk) -> None:
        """Store single chunk"""
        pass
    
    async def store_chunks(self, chunks: List[ContextChunk]) -> None:
        """Store multiple chunks in transaction"""
        pass
    
    async def get_chunks_by_session(self, session_id: str, limit: Optional[int] = None) -> List[ContextChunk]:
        """Retrieve chunks for session, most recent first"""
        pass
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[ContextChunk]:
        """Retrieve specific chunk"""
        pass
    
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks and return count deleted"""
        pass
```

**Required SQL Schema**:
```sql
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    chunk_type TEXT DEFAULT 'conversation',
    entities_json TEXT,
    embedding_json TEXT
);

CREATE INDEX idx_chunks_session ON chunks(session_id);
CREATE INDEX idx_chunks_timestamp ON chunks(timestamp);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Acceptance Criteria**:
- Database schema creates successfully
- All CRUD operations work correctly
- Transactions properly handle errors
- Async/await pattern implemented correctly
- Data integrity maintained across operations

---

### TASK 6: Pressure Relief Valve Implementation
**Objective**: Implement core context pressure relief mechanism
**Files to Create**: `src/virtual_context_mcp/pressure_valve.py`
**Dependencies**: Task 5 complete

**Specifications**:
```python
# Required pressure valve interface
from typing import List, Tuple
from .chunking.chunk import ContextChunk, ContextWindow
from .chunking.tokenizer import TokenCounter, BasicChunker
from .memory.sqlite_store import SQLiteStore

class PressureReliefValve:
    def __init__(self, 
                 token_counter: TokenCounter,
                 chunker: BasicChunker,
                 storage: SQLiteStore,
                 pressure_threshold: float = 0.8,
                 relief_percentage: float = 0.4):
        pass
    
    def calculate_pressure(self, context_window: ContextWindow, max_tokens: int) -> float:
        """Calculate pressure ratio (0.0 to 1.0)"""
        # Return: current_tokens / max_tokens
        pass
    
    async def needs_relief(self, context_window: ContextWindow, max_tokens: int) -> bool:
        """Check if pressure relief is needed"""
        pass
    
    async def execute_relief(self, context_window: ContextWindow, max_tokens: int) -> ContextWindow:
        """Execute pressure relief and return trimmed context"""
        # 1. Calculate tokens to remove (max_tokens * relief_percentage)
        # 2. Select oldest chunks totaling that amount
        # 3. Store selected chunks in SQLite
        # 4. Remove chunks from context window
        # 5. Return updated context window
        pass
    
    def select_chunks_for_relief(self, context_window: ContextWindow, target_tokens: int) -> List[ContextChunk]:
        """Select optimal chunks for removal"""
        # Prefer oldest chunks
        # Try to select complete semantic units
        # Must total approximately target_tokens
        pass
```

**Acceptance Criteria**:
- Pressure calculation is accurate
- Relief triggers only at correct thresholds
- Chunk selection preserves semantic coherence
- Storage operations complete successfully
- Context window size reduced to target level
- No data loss during relief operations

---

### TASK 7: Vector Store Integration (Qdrant)
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

---

### TASK 8: Story Entity Extraction
**Objective**: Extract story-specific entities (characters, locations, plot elements)
**Files to Create**: `src/virtual_context_mcp/entities/story_entities.py`
**Dependencies**: Task 7 complete

**Specifications**:
```python
# Required entity extraction interface
import spacy
import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

@dataclass
class StoryEntity:
    name: str
    entity_type: str  # "character", "location", "plot_element", "theme"
    mentions: List[str]  # Different ways entity is referenced
    context: str  # Surrounding context where found

class StoryEntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy model for NER"""
        pass
    
    def extract_entities(self, text: str) -> Dict[str, List[StoryEntity]]:
        """Extract all story entities from text"""
        # Return format:
        # {
        #     "characters": [StoryEntity(...)],
        #     "locations": [StoryEntity(...)],
        #     "plot_elements": [StoryEntity(...)],
        #     "themes": [StoryEntity(...)]
        # }
        pass
    
    def extract_characters(self, text: str) -> List[StoryEntity]:
        """Extract character names and references"""
        # Use NER for PERSON entities
        # Use dialogue attribution patterns
        # Handle pronouns with coreference resolution
        pass
    
    def extract_locations(self, text: str) -> List[StoryEntity]:
        """Extract location names"""
        # Use NER for GPE, LOC entities
        # Pattern matching for fantasy locations
        pass
    
    def extract_plot_elements(self, text: str) -> List[StoryEntity]:
        """Extract plot-relevant elements"""
        # Look for action verbs, conflicts, resolutions
        # Identify plot devices and story beats
        pass
    
    def extract_dialogue_speakers(self, text: str) -> Dict[str, List[str]]:
        """Map dialogue to speakers"""
        # Parse dialogue attribution
        # Return: {"character_name": ["dialogue1", "dialogue2"]}
        pass
```

**Required Patterns for Story Elements**:
```python
# Pattern examples for implementation
DIALOGUE_PATTERNS = [
    r'"([^"]*)"[,.]?\s*said\s+(\w+)',
    r'(\w+)\s+said[,.]?\s*"([^"]*)"',
    r'"([^"]*)"[,.]?\s*(\w+)\s+replied',
]

LOCATION_MARKERS = [
    r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bthrough\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
]

PLOT_INDICATORS = [
    "decided", "realized", "discovered", "revealed", "confronted",
    "planned", "schemed", "betrayed", "rescued", "defeated"
]
```

**Acceptance Criteria**:
- Character extraction identifies main and minor characters
- Location extraction captures both real and fictional places
- Dialogue attribution works accurately
- Entity deduplication handles variations (John/Johnny/Mr. Smith)
- Plot element extraction identifies story beats
- Performance acceptable for real-time processing

---

### TASK 9: Neo4j Knowledge Graph Integration
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

---

### TASK 10: Memory Retrieval and Relevance Scoring
**Objective**: Implement intelligent memory retrieval combining vector and graph search
**Files to Create**: `src/virtual_context_mcp/retrieval/relevance_scorer.py`
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

---

### TASK 11: Context Manager Core
**Objective**: Implement main context management orchestration
**Files to Create**: `src/virtual_context_mcp/context_manager.py`
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

---

### TASK 12: MCP Server Implementation
**Objective**: Implement MCP protocol server with story-specific tools
**Files to Create**: `src/virtual_context_mcp/server.py`
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

---

### TASK 13: Docker Configuration
**Objective**: Create Docker setup for easy deployment
**Files to Create**: `docker/Dockerfile`, `docker-compose.yml`
**Dependencies**: Task 12 complete

**Specifications**:
```dockerfile
# docker/Dockerfile requirements
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY src/ /app/src/
COPY configs/ /app/configs/

# Install application
RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "virtual_context_mcp", "--config", "configs/novel_writing.yaml"]
```

**Docker Compose Configuration**:
```yaml
# docker-compose.yml requirements
version: '3.8'

services:
  virtual-context:
    build: 
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=storypassword
    depends_on:
      - qdrant
      - neo4j
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333

  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/storypassword
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  qdrant_data:
  neo4j_data:
  neo4j_logs:
```

**Acceptance Criteria**:
- Docker image builds successfully
- All services start with docker-compose up
- Service discovery works between containers
- Volume mounts preserve data correctly
- Port mappings allow external access
- Environment variables override config properly

---

### TASK 14: Integration Testing Suite
**Objective**: Create comprehensive integration tests
**Files to Create**: `tests/integration/test_full_story_workflow.py`
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

---

### TASK 15: CLI and Entry Point
**Objective**: Create command-line interface and package entry point
**Files to Create**: `src/virtual_context_mcp/__main__.py`, `src/virtual_context_mcp/cli.py`
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

---

## LLM Agent Execution Strategy

### Task Dependencies
```
Task 1 (Foundation) → Task 2 (Config) → Task 3 (Models) → Task 4 (Tokenizing)
                                                               ↓
Task 15 (CLI) ← Task 14 (Tests) ← Task 13 (Docker) ← Task 12 (MCP Server)
                                                               ↑
Task 5 (SQLite) → Task 6 (Pressure Relief) → Task 7 (Vector) → Task 8 (Entities)
                                                               ↓
Task 11 (Context Manager) ← Task 10 (Retrieval) ← Task 9 (Neo4j)
```

### Agent Instructions for Each Task

**For optimal LLM agent execution:**

1. **Start each task with**: "I need to implement [Task Name] with the following exact specifications..."

2. **Include in every task prompt**: 
   - The complete interface specification from above
   - All required dependencies
   - Acceptance criteria checklist
   - Error handling requirements

3. **For testing tasks**: Include sample data and expected outputs

4. **For database tasks**: Include exact schema requirements

5. **For async tasks**: Specify proper async/await patterns

### Validation Commands

After each task completion, run these validation commands:

```bash
# Code quality
black src/
isort src/
flake8 src/

# Type checking  
mypy src/

# Unit tests
pytest tests/unit/

# Integration tests (after Task 14)
pytest tests/integration/

# Manual verification
python -m virtual_context_mcp --init-db --config configs/novel_writing.yaml
python -m virtual_context_mcp --debug --config configs/novel_writing.yaml
```

This plan is designed to be executed by LLM coding agents with minimal ambiguity and maximum clarity. Each task is atomic, well-specified, and includes everything needed for successful implementation.