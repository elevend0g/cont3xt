"""Token counting and basic text chunking functionality."""

import re
from typing import List, Tuple

import tiktoken


class TokenCounter:
    """Accurate token counting using tiktoken."""
    
    def __init__(self, model_name: str = "cl100k_base"):
        """Initialize with specific tokenizer model."""
        self.encoding = tiktoken.get_encoding(model_name)
    
    def count_tokens(self, text: str) -> int:
        """Return exact token count for text."""
        return len(self.encoding.encode(text))
    
    def split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks of max_tokens size."""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the split boundary for this chunk
            split_pos = self.find_split_boundary(text[current_pos:], max_tokens)
            if split_pos == 0:  # Safety check to avoid infinite loop
                split_pos = 1
            
            chunk = text[current_pos:current_pos + split_pos]
            chunks.append(chunk)
            current_pos += split_pos
        
        return chunks
    
    def find_split_boundary(self, text: str, target_tokens: int) -> int:
        """Find best character position to split at target_tokens."""
        # Binary search to find the maximum character position that fits in target_tokens
        left, right = 0, len(text)
        best_pos = 0
        
        while left <= right:
            mid = (left + right) // 2
            chunk_text = text[:mid]
            token_count = self.count_tokens(chunk_text)
            
            if token_count <= target_tokens:
                best_pos = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if best_pos == 0:
            return 1  # Ensure we make progress
        
        # Try to find a better boundary (sentence or word)
        text_to_split = text[:best_pos]
        
        # Prefer sentence boundaries
        sentence_endings = ['.', '!', '?', '\n\n']
        for ending in sentence_endings:
            last_sentence = text_to_split.rfind(ending)
            if last_sentence > best_pos * 0.8:  # Within 80% of optimal position
                return last_sentence + 1
        
        # Fall back to word boundaries
        last_space = text_to_split.rfind(' ')
        if last_space > best_pos * 0.8:  # Within 80% of optimal position
            return last_space + 1
        
        # Fall back to newline boundaries
        last_newline = text_to_split.rfind('\n')
        if last_newline > best_pos * 0.8:  # Within 80% of optimal position
            return last_newline + 1
        
        return best_pos


class BasicChunker:
    """Basic text chunking with semantic boundary preservation."""
    
    def __init__(self, token_counter: TokenCounter, chunk_size: int = 3200):
        """Initialize with token counter and target chunk size."""
        self.token_counter = token_counter
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str, preserve_boundaries: bool = True) -> List[str]:
        """Split text into semantic chunks."""
        if not text.strip():
            return []
        
        # If text is already small enough, return as single chunk
        if self.token_counter.count_tokens(text) <= self.chunk_size:
            return [text]
        
        if preserve_boundaries:
            return self._chunk_with_boundaries(text)
        else:
            return self.token_counter.split_by_tokens(text, self.chunk_size)
    
    def _chunk_with_boundaries(self, text: str) -> List[str]:
        """Split text preserving semantic boundaries."""
        # First try to split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if self.token_counter.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Check if single paragraph is too large
                if self.token_counter.count_tokens(paragraph) > self.chunk_size:
                    # Split large paragraph by sentences
                    sentence_chunks = self._split_by_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences when paragraph is too large."""
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.token_counter.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is too large, force split by tokens
                if self.token_counter.count_tokens(sentence) > self.chunk_size:
                    sentence_chunks = self.token_counter.split_by_tokens(sentence, self.chunk_size)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_oldest_chunks(self, chunks: List[str], target_tokens: int) -> List[str]:
        """Extract oldest chunks totaling approximately target_tokens."""
        if not chunks:
            return []
        
        extracted = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.token_counter.count_tokens(chunk)
            
            # If adding this chunk would exceed target, check if we should include it
            if total_tokens + chunk_tokens > target_tokens and extracted:
                # Only include if we're less than halfway to target
                if total_tokens < target_tokens * 0.5:
                    extracted.append(chunk)
                    total_tokens += chunk_tokens
                break
            else:
                extracted.append(chunk)
                total_tokens += chunk_tokens
                
                # If we've reached the target, stop
                if total_tokens >= target_tokens:
                    break
        
        return extracted