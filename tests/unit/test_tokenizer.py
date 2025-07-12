"""
Unit tests for tokenizer functionality
"""

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.virtual_context_mcp.chunking.tokenizer import TokenCounter, BasicChunker


class TestTokenCounter:
    """Test token counting functionality"""
    
    def test_token_counter_initialization(self):
        """Test TokenCounter can be initialized"""
        counter = TokenCounter()
        assert counter is not None
    
    def test_simple_token_counting(self):
        """Test basic token counting"""
        counter = TokenCounter()
        text = "Hello world, this is a test."
        token_count = counter.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 20  # Should be reasonable for short text
    
    def test_empty_text_token_counting(self):
        """Test token counting with empty text"""
        counter = TokenCounter()
        token_count = counter.count_tokens("")
        assert token_count == 0
    
    def test_long_text_token_counting(self):
        """Test token counting with longer text"""
        counter = TokenCounter()
        text = "This is a much longer piece of text that should have more tokens. " * 10
        token_count = counter.count_tokens(text)
        
        assert token_count > 50  # Should have many tokens


class TestBasicChunker:
    """Test basic text chunking functionality"""
    
    def test_chunker_initialization(self):
        """Test BasicChunker can be initialized"""
        chunker = BasicChunker(max_chunk_tokens=1000)
        assert chunker is not None
        assert chunker.max_chunk_tokens == 1000
    
    def test_simple_chunking(self):
        """Test basic text chunking"""
        chunker = BasicChunker(max_chunk_tokens=50)
        text = "This is a test sentence. This is another test sentence. And one more for good measure."
        
        chunks = chunker.chunk_text(text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Verify chunks don't exceed token limit
        for chunk in chunks:
            token_count = chunker.token_counter.count_tokens(chunk)
            assert token_count <= 50
    
    def test_preserve_boundaries(self):
        """Test that sentence boundaries are preserved"""
        chunker = BasicChunker(max_chunk_tokens=100)
        text = "First sentence. Second sentence. Third sentence."
        
        chunks = chunker.chunk_text(text)
        
        # Should preserve sentence endings
        for chunk in chunks:
            if not chunk.endswith('.'):
                # If it doesn't end with period, it should be the last chunk
                # or end with natural boundary
                assert chunk == chunks[-1] or chunk.endswith('\n')
    
    def test_text_reconstruction(self):
        """Test that chunked text can be reconstructed"""
        chunker = BasicChunker(max_chunk_tokens=100)
        original_text = "First paragraph.\n\nSecond paragraph with more text.\n\nThird paragraph."
        
        chunks = chunker.chunk_text(original_text)
        reconstructed = "".join(chunks)
        
        assert reconstructed == original_text