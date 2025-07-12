"""Pressure relief valve for context management."""

from typing import List

from .chunking.chunk import ContextChunk, ContextWindow
from .chunking.tokenizer import TokenCounter, BasicChunker
from .memory.sqlite_store import SQLiteStore


class PressureReliefValve:
    """Manages context pressure by intelligently removing chunks when thresholds are exceeded."""
    
    def __init__(self, 
                 token_counter: TokenCounter,
                 chunker: BasicChunker,
                 storage: SQLiteStore,
                 pressure_threshold: float = 0.8,
                 relief_percentage: float = 0.4):
        """Initialize pressure relief valve with dependencies and thresholds."""
        self.token_counter = token_counter
        self.chunker = chunker
        self.storage = storage
        self.pressure_threshold = pressure_threshold
        self.relief_percentage = relief_percentage
    
    def calculate_pressure(self, context_window: ContextWindow, max_tokens: int) -> float:
        """Calculate pressure ratio (0.0 to 1.0)."""
        if max_tokens <= 0:
            return 1.0
        
        current_tokens = context_window.calculate_tokens()
        return min(current_tokens / max_tokens, 1.0)
    
    async def needs_relief(self, context_window: ContextWindow, max_tokens: int) -> bool:
        """Check if pressure relief is needed."""
        pressure = self.calculate_pressure(context_window, max_tokens)
        return pressure >= self.pressure_threshold
    
    async def execute_relief(self, context_window: ContextWindow, max_tokens: int) -> ContextWindow:
        """Execute pressure relief and return trimmed context."""
        if not await self.needs_relief(context_window, max_tokens):
            return context_window
        
        # 1. Calculate tokens to remove (max_tokens * relief_percentage)
        target_removal_tokens = int(max_tokens * self.relief_percentage)
        
        # 2. Select oldest chunks totaling that amount
        chunks_to_remove = self.select_chunks_for_relief(context_window, target_removal_tokens)
        
        if not chunks_to_remove:
            # No chunks to remove, return original context
            return context_window
        
        # 3. Store selected chunks in SQLite
        await self.storage.store_chunks(chunks_to_remove)
        
        # 4. Remove chunks from context window
        chunk_ids_to_remove = [chunk.id for chunk in chunks_to_remove]
        removed_chunks = context_window.remove_chunks(chunk_ids_to_remove)
        
        # 5. Return updated context window
        return context_window
    
    def select_chunks_for_relief(self, context_window: ContextWindow, target_tokens: int) -> List[ContextChunk]:
        """Select optimal chunks for removal."""
        if not context_window.chunks or target_tokens <= 0:
            return []
        
        # Sort chunks by timestamp (oldest first) to prefer removing oldest chunks
        sorted_chunks = sorted(context_window.chunks, key=lambda chunk: chunk.timestamp)
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            # Check if adding this chunk would exceed target
            if total_tokens + chunk.token_count > target_tokens and selected_chunks:
                # Only include if we're less than halfway to target
                if total_tokens < target_tokens * 0.5:
                    selected_chunks.append(chunk)
                    total_tokens += chunk.token_count
                break
            else:
                selected_chunks.append(chunk)
                total_tokens += chunk.token_count
                
                # If we've reached or exceeded the target, stop
                if total_tokens >= target_tokens:
                    break
        
        return selected_chunks
    
    def get_pressure_stats(self, context_window: ContextWindow, max_tokens: int) -> dict:
        """Get detailed pressure statistics for monitoring."""
        current_tokens = context_window.calculate_tokens()
        pressure = self.calculate_pressure(context_window, max_tokens)
        needs_relief = pressure >= self.pressure_threshold
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "pressure_ratio": pressure,
            "pressure_threshold": self.pressure_threshold,
            "needs_relief": needs_relief,
            "chunk_count": len(context_window.chunks),
            "relief_percentage": self.relief_percentage,
            "tokens_for_relief": int(max_tokens * self.relief_percentage) if needs_relief else 0
        }
    
    async def force_relief(self, context_window: ContextWindow, target_tokens: int) -> ContextWindow:
        """Force pressure relief to specific token count (for emergency situations)."""
        current_tokens = context_window.calculate_tokens()
        
        if current_tokens <= target_tokens:
            return context_window
        
        # Calculate how many tokens to remove
        tokens_to_remove = current_tokens - target_tokens
        
        # Select chunks for removal
        chunks_to_remove = self.select_chunks_for_relief(context_window, tokens_to_remove)
        
        if not chunks_to_remove:
            return context_window
        
        # Store chunks before removal
        await self.storage.store_chunks(chunks_to_remove)
        
        # Remove chunks from context window
        chunk_ids_to_remove = [chunk.id for chunk in chunks_to_remove]
        context_window.remove_chunks(chunk_ids_to_remove)
        
        return context_window