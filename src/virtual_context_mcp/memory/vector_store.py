"""Vector store implementation using Qdrant for semantic similarity search."""

import asyncio
from datetime import datetime
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from ..chunking.chunk import ContextChunk


class VectorStore:
    """Qdrant-based vector store for semantic similarity search."""
    
    def __init__(self, 
                 qdrant_url: str,
                 collection_name: str = "story_chunks",
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Qdrant client and embedding model."""
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()
    
    async def _get_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    # Load model in thread to avoid blocking
                    self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
        return self._model
    
    async def initialize(self) -> None:
        """Initialize the vector store."""
        await self.init_collection()
    
    async def init_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collection_info = self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            vector_config = VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding size
                distance=Distance.COSINE
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config
            )
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = await self._get_model()
        # Generate embedding in thread to avoid blocking
        embedding = await asyncio.to_thread(model.encode, text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def store_chunk(self, chunk: ContextChunk) -> None:
        """Store chunk with embedding in Qdrant."""
        # Generate embedding if not present
        if chunk.embedding is None:
            chunk.embedding = await self.embed_text(chunk.content)
        
        # Prepare point for storage
        point = PointStruct(
            id=chunk.id,
            vector=chunk.embedding,
            payload={
                "session_id": chunk.session_id,
                "content": chunk.content,
                "timestamp": chunk.timestamp.isoformat(),
                "chunk_type": chunk.chunk_type,
                "token_count": chunk.token_count,
                "entities": chunk.entities
            }
        )
        
        # Store in Qdrant
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=[point]
        )
    
    async def search_similar(self, 
                           query_text: str, 
                           session_id: Optional[str] = None,
                           limit: int = 5,
                           score_threshold: float = 0.7) -> List[Tuple[ContextChunk, float]]:
        """Search for similar chunks."""
        # Generate query embedding
        query_embedding = await self.embed_text(query_text)
        
        # Prepare filter
        filter_conditions = None
        if session_id:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                ]
            )
        
        # Search in Qdrant
        search_result = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Convert results to ContextChunk objects
        results = []
        for scored_point in search_result:
            payload = scored_point.payload
            
            # Reconstruct ContextChunk
            chunk = ContextChunk(
                id=scored_point.id,
                session_id=payload["session_id"],
                content=payload["content"],
                token_count=payload["token_count"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                chunk_type=payload.get("chunk_type", "conversation"),
                entities=payload.get("entities"),
                embedding=scored_point.vector
            )
            
            results.append((chunk, scored_point.score))
        
        return results
    
    async def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete chunks from vector store."""
        if not chunk_ids:
            return
        
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=chunk_ids
        )
    
    async def get_collection_info(self) -> dict:
        """Get collection information for monitoring."""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.collection_name
            )
            return {
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def search_by_entities(self, 
                                entity_names: List[str],
                                session_id: Optional[str] = None,
                                limit: int = 10) -> List[Tuple[ContextChunk, float]]:
        """Search for chunks containing specific entities."""
        if not entity_names:
            return []
        
        # Prepare filter conditions
        filter_conditions = []
        if session_id:
            filter_conditions.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id)
                )
            )
        
        # Use scroll to search through all chunks with entity filtering
        scroll_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        scroll_result = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit * 5,  # Get more candidates to filter
            with_payload=True,
            with_vectors=True
        )
        
        results = []
        entity_names_lower = [name.lower() for name in entity_names]
        
        for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
            payload = point.payload
            content = payload.get("content", "").lower()
            entities = payload.get("entities", {})
            
            # Check if any target entities are mentioned
            entity_match = False
            for entity_name in entity_names_lower:
                if entity_name in content:
                    entity_match = True
                    break
                
                # Also check in stored entities
                for entity_type, entity_list in entities.items():
                    if entity_list and any(entity_name in e.lower() for e in entity_list):
                        entity_match = True
                        break
                if entity_match:
                    break
            
            if entity_match:
                chunk = ContextChunk(
                    id=point.id,
                    session_id=payload["session_id"],
                    content=payload["content"],
                    token_count=payload["token_count"],
                    timestamp=datetime.fromisoformat(payload["timestamp"]),
                    chunk_type=payload.get("chunk_type", "conversation"),
                    entities=payload.get("entities"),
                    embedding=point.vector
                )
                
                # Calculate a simple relevance score based on entity matches
                score = 0.5  # Base score for entity match
                results.append((chunk, score))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def search_by_session(self, 
                               session_id: str,
                               limit: int = 100) -> List[ContextChunk]:
        """Get all chunks for a specific session."""
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id)
                )
            ]
        )
        
        # Use scroll to get all points for the session
        scroll_result = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.collection_name,
            scroll_filter=filter_conditions,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        chunks = []
        for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
            payload = point.payload
            
            chunk = ContextChunk(
                id=point.id,
                session_id=payload["session_id"],
                content=payload["content"],
                token_count=payload["token_count"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                chunk_type=payload.get("chunk_type", "conversation"),
                entities=payload.get("entities"),
                embedding=None  # Don't load vectors for this operation
            )
            chunks.append(chunk)
        
        return chunks
    
    async def close(self) -> None:
        """Close Qdrant client connections."""
        if hasattr(self.client, 'close'):
            await asyncio.to_thread(self.client.close)