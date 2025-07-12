"""
Virtual Context MCP Server

A Model Context Protocol server that provides virtual infinite context
for conversational AI systems through intelligent context management.
"""

__version__ = "0.1.0"
__author__ = "Virtual Context Team"
__description__ = "Virtual infinite context MCP server for AI agents"

# Core components
from .context_manager import ContextManager
from .pressure_valve import PressureReliefValve
from .config import Config, ContextConfig, DatabaseConfig
from .chunking import ContextChunk, MemoryEntry, ContextWindow, TokenCounter, BasicChunker
from .memory import SQLiteStore, VectorStore, GraphStore
from .entities import StoryEntity, StoryEntityExtractor
from .retrieval import RelevanceScorer
from .server import VirtualContextMCPServer

__all__ = [
    "ContextManager",
    "PressureReliefValve",
    "Config", 
    "ContextConfig", 
    "DatabaseConfig",
    "ContextChunk", 
    "MemoryEntry", 
    "ContextWindow",
    "TokenCounter", 
    "BasicChunker",
    "SQLiteStore",
    "VectorStore",
    "GraphStore",
    "StoryEntity",
    "StoryEntityExtractor",
    "RelevanceScorer",
    "VirtualContextMCPServer"
]