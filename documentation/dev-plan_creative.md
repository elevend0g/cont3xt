# Virtual Infinite Context MCP Server

## Vision: Infinite Context for Creative Writing

**Primary Use Case**: Enable seamless novel drafting where the AI maintains perfect awareness of characters, plot threads, world-building details, and narrative continuity regardless of context window limitations.

**Core Innovation**: The "context loop" - continuous embedding and memory formation that preserves narrative coherence across unlimited conversation length.

## Revised MVP Architecture

### Full Stack Architecture (Prototype-First)

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client    │───▶│  Context Manager │───▶│ LLM Interface   │
│ (Claude Desktop)│    │     Server       │    │ (Local/API)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                │ Pressure ≥ 80%         │ Response
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Pressure Relief  │    │   Story Memory  │
                       │     Valve        │    │     System      │
                       └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────────────────────────────┐
                       │         Memory Triad                    │
                       │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
                       │  │ SQLite  │ │ Qdrant  │ │  Neo4j   │  │
                       │  │Archive  │ │Vectors  │ │Knowledge │  │
                       │  │         │ │         │ │  Graph   │  │
                       │  └─────────┘ └─────────┘ └──────────┘  │
                       └─────────────────────────────────────────┘
```

### Docker Compose Deployment

**Target Environment**: Ubuntu servers with local databases

```yaml
# docker-compose.yml
version: '3.8'
services:
  context-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URL=bolt://neo4j:7687
    depends_on:
      - qdrant
      - neo4j
    volumes:
      - ./data:/app/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  qdrant_data:
  neo4j_data:
  neo4j_logs:
```

## Prototype Development Strategy

### Phase 1: Core Context Loop (Week 1-2)

**Goal**: Prove the pressure relief valve concept with real creative writing

#### 1.1 Minimal Viable Context Manager

```python
# Core specifications optimized for creative writing
WRITING_CONTEXT_CONFIG = {
    "max_tokens": 12000,           # Working context limit
    "pressure_threshold": 0.8,      # Trigger at 9600 tokens
    "relief_percentage": 0.4,       # Remove 4800 tokens
    "chunk_size": 3200,            # Preserve scene/chapter coherence
    "memory_retrieval_limit": 8,    # Max chunks to retrieve
    "character_boost": 1.5,         # Weight character mentions higher
    "plot_boost": 1.3              # Weight plot elements higher
}
```

#### 1.2 Story-Aware Chunking

```python
class StoryChunker:
    """Intelligent chunking that preserves narrative structure"""
    
    def extract_semantic_chunks(self, context, tokens_to_remove):
        # Identify natural break points
        break_points = self.find_story_breaks(context)
        
        # Prefer chapter/scene boundaries
        # Fall back to paragraph boundaries
        # Preserve dialogue integrity
        
        chunks = self.create_chunks_at_breaks(
            context, 
            break_points, 
            target_size=3200
        )
        
        # Extract oldest chunks while preserving story flow
        return self.select_oldest_complete_chunks(chunks, tokens_to_remove)
    
    def find_story_breaks(self, context):
        """Find natural narrative breakpoints"""
        patterns = [
            r'\n\n---\n\n',        # Scene breaks
            r'\n\n\*\*\*\n\n',     # Chapter breaks  
            r'\n\n(?=[A-Z])',      # New paragraphs
            r'"\s*\n\n',           # End of dialogue
        ]
        # Implementation details...
```

#### 1.3 MCP Interface for Creative Writing

**Resources:**

- `story://current-scene` - Current active narrative context
- `story://characters` - Character registry and traits
- `story://plot-threads` - Active plot lines and status
- `story://world-details` - Setting and world-building info

**Tools:**

- `continue_story` - Add narrative content with context management
- `introduce_character` - Add new character with automatic graph linking
- `develop_plot` - Add plot development with thread tracking
- `describe_setting` - Add world-building with spatial relationships
- `search_story_memory` - Query past narrative elements

### Phase 2: Knowledge Graph Integration (Week 2-3)

#### 2.1 Story Knowledge Graph Schema

```cypher
// Neo4j Schema for Creative Writing
CREATE CONSTRAINT character_name FOR (c:Character) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT location_name FOR (l:Location) REQUIRE l.name IS UNIQUE;
CREATE CONSTRAINT plot_thread_id FOR (p:PlotThread) REQUIRE p.id IS UNIQUE;

// Character relationships
(:Character)-[:KNOWS]->(:Character)
(:Character)-[:LOVES|HATES|FEARS]->(:Character)
(:Character)-[:LOCATED_IN]->(:Location)
(:Character)-[:PARTICIPATES_IN]->(:PlotThread)

// Plot relationships  
(:PlotThread)-[:DEPENDS_ON]->(:PlotThread)
(:PlotThread)-[:CONFLICTS_WITH]->(:PlotThread)
(:PlotThread)-[:OCCURS_IN]->(:Location)

// Narrative structure
(:Scene)-[:FEATURES]->(:Character)
(:Scene)-[:OCCURS_IN]->(:Location)
(:Scene)-[:ADVANCES]->(:PlotThread)
(:Scene)-[:FOLLOWS]->(:Scene)
```

#### 2.2 Entity Extraction for Stories

```python
class StoryEntityExtractor:
    """Extract and classify narrative elements"""
    
    def extract_entities(self, chunk_content):
        entities = {
            'characters': self.extract_characters(chunk_content),
            'locations': self.extract_locations(chunk_content),
            'plot_elements': self.extract_plot_elements(chunk_content),
            'themes': self.extract_themes(chunk_content),
            'timeline_events': self.extract_events(chunk_content)
        }
        
        # Create Neo4j relationships
        relationships = self.infer_relationships(entities)
        
        return entities, relationships
    
    def extract_characters(self, content):
        # Use NER for person detection
        # Pattern matching for dialogue attribution
        # Pronoun resolution for character tracking
        pass
```

### Phase 3: Intelligent Memory Retrieval (Week 3-4)

#### 3.1 Context Assembly for Creative Writing

```python
class StoryContextAssembler:
    """Smart context window composition for narrative consistency"""
    
    async def build_context(self, current_input, session_id):
        # Always include: recent conversation history
        recent_context = await self.get_recent_context(session_id, tokens=6000)
        
        # Identify what the user is working on
        current_focus = await self.analyze_current_focus(current_input)
        
        # Retrieve relevant memories
        relevant_memories = await self.get_relevant_memories(
            current_focus, 
            session_id,
            max_tokens=4000
        )
        
        # Assemble optimal context window
        return self.compose_context_window(
            recent_context, 
            relevant_memories, 
            current_input
        )
    
    async def get_relevant_memories(self, focus_analysis, session_id, max_tokens):
        """Multi-modal memory retrieval"""
        
        # Vector similarity search
        semantic_matches = await self.vector_search(
            focus_analysis.semantic_query,
            limit=5
        )
        
        # Knowledge graph traversal
        if focus_analysis.mentioned_characters:
            graph_matches = await self.graph_search(
                characters=focus_analysis.mentioned_characters,
                relationship_depth=2
            )
        
        # Combine and rank by relevance
        return self.rank_and_limit_memories(
            semantic_matches + graph_matches,
            max_tokens
        )
```

#### 3.2 Creative Writing Memory Scoring

```python
class StoryMemoryScorer:
    """Relevance scoring optimized for narrative consistency"""
    
    def score_memory_relevance(self, memory_chunk, current_context):
        score = 0.0
        
        # Base semantic similarity
        score += self.semantic_similarity(memory_chunk, current_context) * 0.4
        
        # Character consistency boost
        shared_characters = self.get_shared_characters(memory_chunk, current_context)
        score += len(shared_characters) * 0.3
        
        # Plot thread continuity
        shared_plots = self.get_shared_plot_threads(memory_chunk, current_context)
        score += len(shared_plots) * 0.2
        
        # Temporal relevance (recent events matter more)
        score += self.temporal_relevance(memory_chunk) * 0.1
        
        return score
```

### Phase 4: LLM Integration & Testing (Week 4-5)

#### 4.1 Multi-Model Support

```python
class LLMInterface:
    """Unified interface for local and API models"""
    
    def __init__(self, config):
        self.config = config
        
        if config.model_type == "local":
            self.client = self.setup_local_model(config.model_path)
        elif config.model_type == "api":
            self.client = self.setup_api_client(config.api_config)
    
    async def generate_response(self, context_window, generation_params):
        """Generate response with optimal context"""
        
        # Ensure context fits within model limits
        optimized_context = self.optimize_context_for_model(context_window)
        
        # Generate with model-specific parameters
        response = await self.client.generate(
            prompt=optimized_context.to_prompt(),
            **generation_params
        )
        
        return response
```

#### 4.2 Creative Writing Test Suite

```python
# Test scenarios for novel writing
TEST_SCENARIOS = [
    {
        "name": "character_consistency",
        "description": "Maintain character traits across 50+ interactions",
        "setup": "Introduce 5 main characters with distinct personalities",
        "test": "Reference character traits established 40 interactions ago"
    },
    {
        "name": "plot_continuity", 
        "description": "Track multiple plot threads simultaneously",
        "setup": "Establish 3 interconnected plot lines",
        "test": "Weave plot threads together after context relief cycles"
    },
    {
        "name": "world_building_persistence",
        "description": "Maintain consistent world details",
        "setup": "Create detailed fantasy world with geography/history",
        "test": "Reference world details established earlier in long conversation"
    }
]
```

## Implementation Priorities

### Week 1: Foundation + Pressure Relief

- [ ] Basic MCP server with Docker deployment
- [ ] Pressure relief valve implementation
- [ ] SQLite storage for conversation archival
- [ ] Simple chunking based on narrative breaks

### Week 2: Knowledge Graph Core

- [ ] Neo4j integration with story schema
- [ ] Basic entity extraction (characters, locations)
- [ ] Relationship inference and storage
- [ ] Graph-based memory retrieval

### Week 3: Vector Memory

- [ ] Qdrant integration for semantic search
- [ ] Story-aware embedding strategies
- [ ] Multi-modal memory retrieval system
- [ ] Context assembly optimization

### Week 4: Creative Writing Tools

- [ ] Story-specific MCP tools implementation
- [ ] Character/plot/setting management
- [ ] Memory relevance scoring for narratives
- [ ] Claude Desktop integration testing

### Week 5: Polish & Real-World Testing

- [ ] LLM interface for local and API models
- [ ] Performance optimization
- [ ] Creative writing test scenarios
- [ ] Documentation and setup guides

## Success Criteria for Novel Writing

### Functional Goals

1. **Seamless Context Transitions**: No interruption to creative flow during pressure relief
2. **Character Consistency**: AI maintains character voice and traits across entire novel
3. **Plot Coherence**: Complex plot threads remain consistent and interconnected
4. **World-Building Persistence**: Setting details remain consistent throughout

### Technical Goals

1. **Context Relief Efficiency**: Relief operations complete in <500ms
2. **Memory Precision**: Relevant past narrative elements surface automatically
3. **Relationship Tracking**: Knowledge graph captures character/plot relationships
4. **Multi-Model Compatibility**: Works with both local LLMs and API models

### User Experience Goals

1. **Invisible Operation**: User never needs to manually manage context
2. **Enhanced Creativity**: AI suggestions improve due to perfect story memory
3. **No Repetition**: System prevents rehashing of previously established elements
4. **Narrative Flow**: Maintains natural story progression despite technical complexity

## Development Environment Setup

### Server Requirements

- Ubuntu 20.04+ with Docker and Docker Compose
- 16GB+ RAM recommended for local LLM testing
- SSD storage for database performance
- GPU optional but recommended for local embedding generation

### Quick Start Commands

```bash
git clone <repo>
cd virtual-context-mcp
docker-compose up -d
pip install -e .
python -m virtual_context_mcp --config novel_writing.yaml
```

### Claude Desktop Integration

```json
// Claude Desktop MCP config
{
  "mcpServers": {
    "virtual-context": {
      "command": "python",
      "args": ["-m", "virtual_context_mcp"],
      "env": {
        "CONTEXT_CONFIG": "novel_writing"
      }
    }
  }
}
```

This revised plan prioritizes your creative writing use case while building the full knowledge graph capabilities from the start. The prototype-first approach means we can iterate quickly on the core concept and refine based on real novel-writing sessions.

Ready to start with the foundation setup, or would you like to adjust any specific aspects of this approach?