"""Core data models for chunks and memory entries."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """A chunk of context with metadata for storage and retrieval."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: str
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.now)
    chunk_type: str = "conversation"  # conversation, character, plot, setting
    entities: Optional[Dict[str, List[str]]] = None
    embedding: Optional[List[float]] = None
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize for storage."""
        data = self.model_dump()
        # Convert datetime to ISO string for storage
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ContextChunk":
        """Deserialize from storage."""
        # Convert ISO string back to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class MemoryEntry(BaseModel):
    """A memory entry with relevance scoring for retrieval."""
    
    chunk_id: str
    relevance_score: float
    retrieval_reason: str  # "semantic", "graph", "temporal"


class ContextWindow(BaseModel):
    """A window of context chunks with token management."""
    
    chunks: List[ContextChunk]
    total_tokens: int
    session_id: str
    
    def calculate_tokens(self) -> int:
        """Calculate total token count from all chunks."""
        total = sum(chunk.token_count for chunk in self.chunks)
        self.total_tokens = total
        return total
    
    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add chunk and update token count."""
        self.chunks.append(chunk)
        self.total_tokens += chunk.token_count
    
    def remove_chunks(self, chunk_ids: List[str]) -> List[ContextChunk]:
        """Remove chunks by IDs and return them."""
        removed_chunks = []
        remaining_chunks = []
        
        for chunk in self.chunks:
            if chunk.id in chunk_ids:
                removed_chunks.append(chunk)
                self.total_tokens -= chunk.token_count
            else:
                remaining_chunks.append(chunk)
        
        self.chunks = remaining_chunks
        return removed_chunks