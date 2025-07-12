"""
Memory retrieval and relevance scoring system.

This module implements intelligent memory retrieval combining vector and graph search
with multi-factor relevance scoring for optimal context reconstruction.
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set

from ..chunking.chunk import ContextChunk, MemoryEntry
from ..memory.vector_store import VectorStore
from ..memory.graph_store import GraphStore
from ..entities.story_entities import StoryEntityExtractor


# Required scoring weights and calculations
RELEVANCE_WEIGHTS = {
    "semantic": 0.4,
    "entity_overlap": 0.3,
    "temporal": 0.2,
    "graph_connectivity": 0.1
}


class RelevanceScorer:
    """
    Intelligent memory retrieval system with multi-factor relevance scoring.
    
    Combines semantic similarity, entity overlap, temporal relevance, and graph
    connectivity to identify the most relevant memories for current context.
    """
    
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        """Initialize the relevance scorer with storage systems."""
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.entity_extractor = StoryEntityExtractor()
        self._lock = asyncio.Lock()
    
    async def score_memory_relevance(self, 
                                   memory_chunk: ContextChunk, 
                                   current_context: str,
                                   session_id: str) -> float:
        """
        Calculate relevance score for memory chunk.
        
        Combines multiple scoring factors:
        - Semantic similarity (40% weight)
        - Entity overlap (30% weight)
        - Temporal relevance (20% weight)
        - Graph connectivity (10% weight)
        
        Args:
            memory_chunk: The chunk to score
            current_context: Current input text for comparison
            session_id: Session identifier for context
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Calculate individual scoring components
        semantic_score = await self.calculate_semantic_similarity(
            memory_chunk.content, current_context
        )
        
        entity_overlap_score = await self.calculate_entity_overlap(
            memory_chunk, current_context
        )
        
        temporal_score = self.calculate_temporal_relevance(memory_chunk)
        
        # Extract entities from current context for graph scoring
        current_entities = await self._extract_entity_names(current_context)
        graph_score = await self.calculate_graph_connectivity(
            memory_chunk, current_entities, session_id
        )
        
        # Combine scores with weights
        total_score = (
            RELEVANCE_WEIGHTS["semantic"] * semantic_score +
            RELEVANCE_WEIGHTS["entity_overlap"] * entity_overlap_score +
            RELEVANCE_WEIGHTS["temporal"] * temporal_score +
            RELEVANCE_WEIGHTS["graph_connectivity"] * graph_score
        )
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, total_score))
    
    async def get_relevant_memories(self, 
                                  current_input: str,
                                  session_id: str,
                                  max_memories: int = 8,
                                  min_score: float = 0.3) -> List[MemoryEntry]:
        """
        Get most relevant memories for current input.
        
        Args:
            current_input: Current user input text
            session_id: Session identifier
            max_memories: Maximum number of memories to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of memory entries ranked by relevance score
        """
        # 1. Vector search for semantic similarity
        similar_chunks = await self.vector_store.search_similar(
            current_input, session_id, limit=max_memories * 3  # Get more candidates
        )
        
        # 2. Extract entities from current input
        current_entities = await self._extract_entity_names(current_input)
        
        # 3. Graph search for connected entities (if we have entities)
        graph_chunks = []
        if current_entities:
            # Get chunks that contain entities connected to current entities
            connected_entities = await self.graph_store.get_connected_entities(
                current_entities, session_id, max_depth=2
            )
            if connected_entities:
                graph_chunks = await self.vector_store.search_by_entities(
                    connected_entities, session_id, limit=max_memories * 2
                )
        
        # 4. Combine and deduplicate candidates
        all_candidates = {}
        for chunk, _ in similar_chunks:
            all_candidates[chunk.id] = chunk
        for chunk, _ in graph_chunks:
            all_candidates[chunk.id] = chunk
        
        # 5. Score and rank all candidate memories
        scored_memories = []
        for chunk in all_candidates.values():
            relevance_score = await self.score_memory_relevance(
                chunk, current_input, session_id
            )
            
            if relevance_score >= min_score:
                # Determine primary retrieval reason
                semantic_score = await self.calculate_semantic_similarity(
                    chunk.content, current_input
                )
                entity_overlap_score = await self.calculate_entity_overlap(
                    chunk, current_input
                )
                
                if semantic_score > 0.7:
                    reason = "semantic"
                elif entity_overlap_score > 0.5:
                    reason = "entity_overlap" 
                else:
                    reason = "graph_connectivity"
                
                memory_entry = MemoryEntry(
                    chunk_id=chunk.id,
                    relevance_score=relevance_score,
                    retrieval_reason=reason
                )
                scored_memories.append(memory_entry)
        
        # 6. Sort by relevance score and return top memories
        scored_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        return scored_memories[:max_memories]
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between texts using vector embeddings.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Use vector store's embedding model for consistency
            model = await self.vector_store._get_model()
            
            # Generate embeddings for both texts
            async with self._lock:
                embeddings = model.encode([text1, text2])
                embedding1 = embeddings[0]
                embedding2 = embeddings[1]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            magnitude1 = math.sqrt(sum(a * a for a in embedding1))
            magnitude2 = math.sqrt(sum(a * a for a in embedding2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            
            # Convert from [-1, 1] to [0, 1] range
            return (cosine_similarity + 1.0) / 2.0
            
        except Exception:
            # Fallback to simple text similarity if embedding fails
            return self._simple_text_similarity(text1, text2)
    
    async def calculate_entity_overlap(self, chunk: ContextChunk, current_text: str) -> float:
        """
        Calculate entity overlap score using Jaccard similarity.
        
        Args:
            chunk: Memory chunk to compare
            current_text: Current input text
            
        Returns:
            Entity overlap score between 0.0 and 1.0
        """
        # Extract entities from chunk (use cached if available)
        chunk_entities = set()
        if chunk.entities:
            for entity_type, entities in chunk.entities.items():
                chunk_entities.update(entities)
        else:
            # Extract entities if not cached
            extracted = self.entity_extractor.extract_entities(chunk.content)
            for entity in extracted:
                chunk_entities.add(entity.name.lower())
        
        # Extract entities from current text
        current_entities = await self._extract_entity_names(current_text)
        current_entity_set = set(entity.lower() for entity in current_entities)
        
        # Calculate Jaccard similarity
        if not chunk_entities and not current_entity_set:
            return 0.0
        
        intersection = len(chunk_entities.intersection(current_entity_set))
        union = len(chunk_entities.union(current_entity_set))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_temporal_relevance(self, chunk: ContextChunk) -> float:
        """
        Calculate temporal relevance with exponential decay.
        
        Args:
            chunk: Memory chunk to score
            
        Returns:
            Temporal relevance score between 0.0 and 1.0
        """
        # Calculate age in hours
        age_delta = datetime.now() - chunk.timestamp
        age_hours = age_delta.total_seconds() / 3600
        
        # Exponential decay with 24-hour half-life
        # score = exp(-age_hours / 24)
        temporal_score = math.exp(-age_hours / 24.0)
        
        return temporal_score
    
    async def calculate_graph_connectivity(self, 
                                         chunk: ContextChunk, 
                                         current_entities: List[str], 
                                         session_id: str) -> float:
        """
        Calculate graph connectivity score based on entity relationships.
        
        Args:
            chunk: Memory chunk to analyze
            current_entities: Entities from current input
            session_id: Session identifier
            
        Returns:
            Graph connectivity score between 0.0 and 1.0
        """
        if not current_entities:
            return 0.0
        
        try:
            # Get entities from chunk
            chunk_entities = []
            if chunk.entities:
                for entity_type, entities in chunk.entities.items():
                    chunk_entities.extend(entities)
            else:
                # Extract entities if not cached
                extracted = self.entity_extractor.extract_entities(chunk.content)
                chunk_entities = [entity.name for entity in extracted]
            
            if not chunk_entities:
                return 0.0
            
            # Count shared relationships between chunk entities and current entities
            shared_relationships = 0
            total_possible = len(chunk_entities) * len(current_entities)
            
            for chunk_entity in chunk_entities:
                for current_entity in current_entities:
                    # Check if entities are connected in the graph
                    connected = await self.graph_store.are_entities_connected(
                        chunk_entity, current_entity, session_id, max_depth=3
                    )
                    if connected:
                        shared_relationships += 1
            
            # Calculate connectivity score
            connectivity = shared_relationships / total_possible if total_possible > 0 else 0.0
            return connectivity
            
        except Exception:
            # Return 0.0 if graph operations fail
            return 0.0
    
    async def _extract_entity_names(self, text: str) -> List[str]:
        """Extract entity names from text."""
        entities = self.entity_extractor.extract_entities(text)
        return [entity.name for entity in entities]
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple fallback text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0