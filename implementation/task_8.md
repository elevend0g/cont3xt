### TASK 8: Story Entity Extraction

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