"""Core context management orchestration."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .chunking.chunk import ContextChunk, ContextWindow, MemoryEntry
from .chunking.tokenizer import TokenCounter, BasicChunker
from .pressure_valve import PressureReliefValve
from .memory.sqlite_store import SQLiteStore
from .memory.vector_store import VectorStore
from .memory.graph_store import GraphStore
from .retrieval.relevance_scorer import RelevanceScorer
from .entities.story_entities import StoryEntityExtractor
from .config.settings import Config

logger = logging.getLogger(__name__)


class ContextManager:
    """Core orchestrator for virtual infinite context management."""
    
    def __init__(self, config: Config):
        """Initialize all components."""
        self.config = config
        
        # Core components
        self.token_counter = TokenCounter(model_name=config.context.token_model)
        self.chunker = BasicChunker(
            token_counter=self.token_counter,
            chunk_size=config.context.chunk_size
        )
        
        # Storage systems
        self.sqlite_store = SQLiteStore(config.database.sqlite_path)
        self.vector_store = VectorStore(
            qdrant_url=config.database.qdrant_url,
            collection_name=config.database.qdrant_collection
        )
        self.graph_store = GraphStore(
            uri=config.database.neo4j_url,
            user=config.database.neo4j_user,
            password=config.database.neo4j_password
        )
        
        # Processing components
        self.entity_extractor = StoryEntityExtractor()
        self.pressure_valve = PressureReliefValve(
            token_counter=self.token_counter,
            chunker=self.chunker,
            storage=self.sqlite_store,
            pressure_threshold=config.context.pressure_threshold,
            relief_percentage=config.context.relief_percentage
        )
        self.relevance_scorer = RelevanceScorer(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            sqlite_store=self.sqlite_store
        )
        
        # Session state tracking
        self._current_windows: Dict[str, ContextWindow] = {}
        self._last_relief: Dict[str, datetime] = {}
        
    async def initialize(self) -> None:
        """Initialize databases and connections."""
        logger.info("Initializing ContextManager components...")
        
        # Initialize all storage systems
        await self.sqlite_store.initialize()
        await self.vector_store.initialize()
        await self.graph_store.initialize()
        
        # Initialize entity extractor
        await self.entity_extractor.initialize()
        
        logger.info("ContextManager initialization complete")
        
    async def close(self) -> None:
        """Clean up resources."""
        await self.sqlite_store.close()
        await self.vector_store.close()
        await self.graph_store.close()
        
    async def process_interaction(self, 
                                user_input: str, 
                                assistant_response: str,
                                session_id: str) -> ContextWindow:
        """Process a complete user-assistant interaction."""
        logger.debug(f"Processing interaction for session {session_id}")
        
        # 1. Create chunks from interaction
        interaction_text = f"User: {user_input}\n\nAssistant: {assistant_response}"
        chunks = await self._create_chunks_from_text(interaction_text, session_id)
        
        # 2. Add to current context window
        current_window = self._current_windows.get(session_id, ContextWindow(
            chunks=[], 
            session_id=session_id
        ))
        current_window.chunks.extend(chunks)
        
        # 3. Check pressure and trigger relief if needed
        pressure = self._calculate_pressure(current_window)
        if pressure > self.config.context.pressure_threshold:
            logger.info(f"Pressure threshold exceeded ({pressure:.2f}), triggering relief")
            await self._trigger_relief(current_window)
            self._last_relief[session_id] = datetime.utcnow()
        
        # 4. Store interaction in memory systems (async, non-blocking)
        asyncio.create_task(self._store_interaction_async(chunks, session_id))
        
        # 5. Update and return context window
        self._current_windows[session_id] = current_window
        return current_window
        
    async def build_context_window(self, 
                                 current_input: str,
                                 session_id: str,
                                 max_tokens: int) -> ContextWindow:
        """Build optimal context window for LLM inference."""
        logger.debug(f"Building context window for session {session_id}, max_tokens={max_tokens}")
        
        # 1. Reserve tokens for current input (actual + buffer)
        input_tokens = self.token_counter.count_tokens(current_input)
        available_tokens = max_tokens - input_tokens - 500  # 500 token buffer
        
        if available_tokens <= 0:
            logger.warning(f"Input too large ({input_tokens} tokens), returning minimal context")
            return ContextWindow(chunks=[], session_id=session_id)
        
        # 2. Get recent conversation (70% of available tokens)
        recent_tokens = int(available_tokens * 0.7)
        recent_chunks = await self._get_recent_chunks(session_id, recent_tokens)
        
        # 3. Get relevant memories (30% of available tokens)
        used_tokens = sum(chunk.token_count for chunk in recent_chunks)
        memory_tokens = available_tokens - used_tokens
        
        relevant_memories = []
        if memory_tokens > 0:
            relevant_memories = await self.relevance_scorer.get_relevant_memories(
                current_input, session_id, max_tokens=memory_tokens
            )
        
        # 4. Combine and create context window
        all_chunks = recent_chunks + [entry.chunk for entry in relevant_memories]
        
        # Ensure we don't exceed token limit
        final_chunks = self._trim_chunks_to_limit(all_chunks, available_tokens)
        
        context_window = ContextWindow(chunks=final_chunks, session_id=session_id)
        logger.debug(f"Built context window with {len(final_chunks)} chunks, "
                    f"{context_window.total_tokens} tokens")
        
        return context_window
        
    async def add_interaction(self, 
                            user_input: str, 
                            assistant_response: str,
                            session_id: str) -> None:
        """Add new interaction to context."""
        await self.process_interaction(user_input, assistant_response, session_id)
        
    async def get_current_pressure(self, session_id: str) -> float:
        """Get current context pressure for session."""
        window = self._current_windows.get(session_id)
        if not window:
            return 0.0
        return self._calculate_pressure(window)
        
    async def force_relief(self, session_id: str) -> int:
        """Force context relief and return tokens removed."""
        window = self._current_windows.get(session_id)
        if not window:
            return 0
            
        initial_tokens = window.total_tokens
        await self._trigger_relief(window)
        self._last_relief[session_id] = datetime.utcnow()
        
        tokens_removed = initial_tokens - window.total_tokens
        logger.info(f"Force relief removed {tokens_removed} tokens from session {session_id}")
        return tokens_removed
        
    async def get_context_stats(self, session_id: str) -> Dict[str, Any]:
        """Get context statistics for session."""
        window = self._current_windows.get(session_id)
        
        if not window:
            return {
                "current_tokens": 0,
                "pressure": 0.0,
                "total_chunks": 0,
                "last_relief": None,
                "max_tokens": self.config.context.max_tokens
            }
            
        return {
            "current_tokens": window.total_tokens,
            "pressure": self._calculate_pressure(window),
            "total_chunks": len(window.chunks),
            "last_relief": self._last_relief.get(session_id),
            "max_tokens": self.config.context.max_tokens,
            "pressure_threshold": self.config.context.pressure_threshold,
            "relief_percentage": self.config.context.relief_percentage
        }
        
    # Private helper methods
    
    async def _create_chunks_from_text(self, text: str, session_id: str) -> List[ContextChunk]:
        """Create chunks from text with metadata."""
        chunks = await self.chunker.chunk_text(text)
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.session_id = session_id
            chunk.timestamp = datetime.utcnow()
            
        return chunks
        
    def _calculate_pressure(self, window: ContextWindow) -> float:
        """Calculate pressure for context window."""
        return self.pressure_valve.calculate_pressure(window)
        
    async def _trigger_relief(self, window: ContextWindow) -> None:
        """Trigger pressure relief on context window."""
        await self.pressure_valve.relieve_pressure(window)
        
    async def _get_recent_chunks(self, session_id: str, max_tokens: int) -> List[ContextChunk]:
        """Get recent chunks from current session."""
        current_window = self._current_windows.get(session_id)
        if not current_window:
            return []
            
        # Get chunks in reverse chronological order
        sorted_chunks = sorted(current_window.chunks, key=lambda c: c.timestamp, reverse=True)
        
        # Take chunks until we hit token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if total_tokens + chunk.token_count <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk.token_count
            else:
                break
                
        # Return in chronological order
        return sorted(selected_chunks, key=lambda c: c.timestamp)
        
    def _trim_chunks_to_limit(self, chunks: List[ContextChunk], max_tokens: int) -> List[ContextChunk]:
        """Trim chunks to fit within token limit."""
        total_tokens = 0
        selected_chunks = []
        
        for chunk in chunks:
            if total_tokens + chunk.token_count <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk.token_count
            else:
                break
                
        return selected_chunks
        
    async def _store_interaction_async(self, chunks: List[ContextChunk], session_id: str) -> None:
        """Store interaction in all memory systems asynchronously."""
        try:
            # Extract entities from chunks
            entities = []
            for chunk in chunks:
                chunk_entities = await self.entity_extractor.extract_entities(chunk.content)
                entities.extend(chunk_entities)
            
            # Store in parallel
            await asyncio.gather(
                self._store_chunks_sqlite(chunks),
                self._store_chunks_vector(chunks),
                self._store_entities_graph(entities),
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            
    async def _store_chunks_sqlite(self, chunks: List[ContextChunk]) -> None:
        """Store chunks in SQLite."""
        for chunk in chunks:
            await self.sqlite_store.store_chunk(chunk)
            
    async def _store_chunks_vector(self, chunks: List[ContextChunk]) -> None:
        """Store chunks in vector store."""
        for chunk in chunks:
            await self.vector_store.store_chunk(chunk)
            
    async def _store_entities_graph(self, entities: List) -> None:
        """Store entities in graph store."""
        for entity in entities:
            await self.graph_store.store_entity(entity)