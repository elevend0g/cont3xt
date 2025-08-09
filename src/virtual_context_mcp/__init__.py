"""
Virtual Context MCP Server

A Model Context Protocol server that provides virtual infinite context
for conversational AI systems through intelligent context management.
"""

__version__ = "0.1.1"
__author__ = "Virtual Context Team"
__description__ = "Virtual infinite context MCP server for AI agents"

# Public API (MVP-friendly; avoid importing heavy optional modules)
try:
    from .server import VirtualContextMCPServer
except Exception:
    VirtualContextMCPServer = None  # type: ignore

__all__ = [
    "VirtualContextMCPServer",
    "__version__",
    "__author__",
    "__description__",
]
