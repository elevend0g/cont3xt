"""
MCP Server Implementation for Virtual Infinite Context

This module implements the Model Context Protocol (MCP) server interface for the
virtual infinite context system, providing story-specific tools and resources.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from mcp import McpError
    from mcp.server import Server
    from mcp.types import (
        Resource, 
        Tool, 
        TextContent, 
        CallToolResult,
        ListResourcesResult,
        ReadResourceResult
    )
except ImportError:
    # Fallback for development
    class McpError(Exception):
        pass
    
    class Server:
        def __init__(self, name: str):
            self.name = name
            self._tools = {}
            self._resources = {}
        
        def call_tool(self):
            def decorator(func):
                self._tools[func.__name__] = func
                return func
            return decorator
        
        def list_resources(self):
            def decorator(func):
                self._resources['list'] = func
                return func
            return decorator
        
        def read_resource(self):
            def decorator(func):
                self._resources['read'] = func
                return func
            return decorator
    
    class Resource:
        def __init__(self, uri: str, name: str, description: str, mimeType: str):
            self.uri = uri
            self.name = name
            self.description = description
            self.mimeType = mimeType
    
    class Tool:
        def __init__(self, name: str, description: str, inputSchema: Dict[str, Any]):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text
    
    class CallToolResult:
        def __init__(self, content: List[TextContent]):
            self.content = content
    
    class ListResourcesResult:
        def __init__(self, resources: List[Resource]):
            self.resources = resources
    
    class ReadResourceResult:
        def __init__(self, contents: List[TextContent]):
            self.contents = contents

from .context_manager import ContextManager
from .config.settings import Config, load_config

logger = logging.getLogger(__name__)


class VirtualContextMCPServer:
    """MCP Server for Virtual Infinite Context system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP server with context manager"""
        self.config = load_config(config_path)
        self.context_manager: Optional[ContextManager] = None
        self.server: Optional[Server] = None
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize the context manager and server components"""
        try:
            # Initialize context manager
            self.context_manager = ContextManager(self.config)
            await self.context_manager.initialize()
            
            logger.info("Virtual Context MCP Server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize context manager: {e}")
            raise
    
    async def setup_server(self) -> Server:
        """Configure MCP server with tools and resources"""
        if self.context_manager is None:
            await self.initialize()
        
        server = Server("virtual-context")
        self.server = server
        
        # Register tools
        self._register_continue_story_tool()
        self._register_search_story_memory_tool()
        self._register_get_character_info_tool()
        self._register_get_plot_threads_tool()
        self._register_get_context_stats_tool()
        
        # Register resources
        self._register_resources()
        
        logger.info("MCP server setup completed with all tools and resources")
        return server
    
    def _register_continue_story_tool(self) -> None:
        """Register the continue_story tool"""
        
        @self.server.call_tool()
        async def continue_story(arguments: dict) -> CallToolResult:
            """Add story content and manage context automatically"""
            try:
                # Validate required arguments
                required_args = ['user_input', 'assistant_response', 'session_id']
                for arg in required_args:
                    if arg not in arguments:
                        raise McpError(f"Missing required argument: {arg}")
                
                user_input = arguments['user_input']
                assistant_response = arguments['assistant_response']
                session_id = arguments['session_id']
                
                # Process the interaction through context manager
                result = await self.context_manager.process_interaction(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    session_id=session_id
                )
                
                # Get current context window and pressure status
                context_window = await self.context_manager.build_context_window(session_id)
                pressure = self.context_manager.calculate_pressure(context_window)
                
                response_data = {
                    "status": "success",
                    "context_window_summary": {
                        "total_tokens": context_window.total_tokens,
                        "chunk_count": len(context_window.chunks),
                        "pressure_level": f"{pressure:.1%}"
                    },
                    "pressure_status": {
                        "current_pressure": pressure,
                        "threshold": self.config.context.pressure_threshold,
                        "needs_relief": pressure > self.config.context.pressure_threshold
                    },
                    "processing_result": result
                }
                
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(response_data, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error in continue_story: {e}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ])
    
    def _register_search_story_memory_tool(self) -> None:
        """Register the search_story_memory tool"""
        
        @self.server.call_tool()
        async def search_story_memory(arguments: dict) -> CallToolResult:
            """Search story memory for relevant details"""
            try:
                # Validate required arguments
                if 'query' not in arguments or 'session_id' not in arguments:
                    raise McpError("Missing required arguments: query, session_id")
                
                query = arguments['query']
                session_id = arguments['session_id']
                entity_type = arguments.get('entity_type')
                max_results = arguments.get('max_results', 10)
                
                # Use the retrieval system to find relevant memories
                memories = await self.context_manager.retrieval_system.get_relevant_memories(
                    query=query,
                    session_id=session_id,
                    max_results=max_results
                )
                
                # Filter by entity type if specified
                if entity_type:
                    memories = [
                        memory for memory in memories
                        if any(entity.entity_type == entity_type for entity in memory['chunk'].entities)
                    ]
                
                response_data = {
                    "status": "success",
                    "query": query,
                    "results_count": len(memories),
                    "memories": [
                        {
                            "chunk_id": memory['chunk'].chunk_id,
                            "content": memory['chunk'].content,
                            "relevance_score": memory['score'],
                            "timestamp": memory['chunk'].timestamp.isoformat(),
                            "entities": [
                                {
                                    "name": entity.name,
                                    "type": entity.entity_type,
                                    "confidence": entity.confidence
                                }
                                for entity in memory['chunk'].entities
                            ]
                        }
                        for memory in memories
                    ]
                }
                
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(response_data, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error in search_story_memory: {e}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ])
    
    def _register_get_character_info_tool(self) -> None:
        """Register the get_character_info tool"""
        
        @self.server.call_tool()
        async def get_character_info(arguments: dict) -> CallToolResult:
            """Get information about story characters"""
            try:
                # Validate required arguments
                if 'character_name' not in arguments or 'session_id' not in arguments:
                    raise McpError("Missing required arguments: character_name, session_id")
                
                character_name = arguments['character_name']
                session_id = arguments['session_id']
                
                # Search for character-related memories
                character_memories = await self.context_manager.retrieval_system.get_relevant_memories(
                    query=character_name,
                    session_id=session_id,
                    max_results=20
                )
                
                # Filter memories that contain the character
                character_chunks = []
                for memory in character_memories:
                    chunk = memory['chunk']
                    for entity in chunk.entities:
                        if (entity.entity_type == 'character' and 
                            character_name.lower() in entity.name.lower()):
                            character_chunks.append({
                                'chunk': chunk,
                                'score': memory['score'],
                                'entity': entity
                            })
                            break
                
                # Get character relationships from graph store
                relationships = []
                try:
                    connected_entities = await self.context_manager.graph_store.get_connected_entities(
                        entity_name=character_name,
                        entity_type='character',
                        session_id=session_id
                    )
                    relationships = [
                        {
                            "name": entity['name'],
                            "type": entity['type'],
                            "relationship": entity.get('relationship', 'appears_with')
                        }
                        for entity in connected_entities
                    ]
                except Exception as e:
                    logger.warning(f"Could not retrieve character relationships: {e}")
                
                response_data = {
                    "status": "success",
                    "character_name": character_name,
                    "mentions_count": len(character_chunks),
                    "character_details": {
                        "recent_mentions": [
                            {
                                "content": chunk_data['chunk'].content,
                                "timestamp": chunk_data['chunk'].timestamp.isoformat(),
                                "relevance_score": chunk_data['score']
                            }
                            for chunk_data in character_chunks[:5]  # Most recent 5
                        ],
                        "attributes": [
                            chunk_data['entity'].attributes
                            for chunk_data in character_chunks
                            if chunk_data['entity'].attributes
                        ][:10]  # Top 10 unique attributes
                    },
                    "relationships": relationships
                }
                
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(response_data, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error in get_character_info: {e}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ])
    
    def _register_get_plot_threads_tool(self) -> None:
        """Register the get_plot_threads tool"""
        
        @self.server.call_tool()
        async def get_plot_threads(arguments: dict) -> CallToolResult:
            """Get active plot threads and their status"""
            try:
                # Validate required arguments
                if 'session_id' not in arguments:
                    raise McpError("Missing required argument: session_id")
                
                session_id = arguments['session_id']
                
                # Get all plot elements from recent memory
                plot_memories = await self.context_manager.retrieval_system.get_relevant_memories(
                    query="plot conflict resolution action",
                    session_id=session_id,
                    max_results=50
                )
                
                # Extract plot elements
                plot_elements = []
                for memory in plot_memories:
                    chunk = memory['chunk']
                    for entity in chunk.entities:
                        if entity.entity_type == 'plot_element':
                            plot_elements.append({
                                'name': entity.name,
                                'content': chunk.content,
                                'timestamp': chunk.timestamp,
                                'attributes': entity.attributes,
                                'score': memory['score']
                            })
                
                # Group by status/resolution
                active_plots = []
                resolved_plots = []
                
                for plot in plot_elements:
                    # Simple heuristic: look for resolution keywords in recent content
                    content_lower = plot['content'].lower()
                    if any(word in content_lower for word in ['resolved', 'concluded', 'ended', 'finished']):
                        resolved_plots.append(plot)
                    else:
                        active_plots.append(plot)
                
                # Get plot connections from graph
                plot_connections = []
                try:
                    for plot in active_plots[:10]:  # Top 10 active plots
                        connected = await self.context_manager.graph_store.get_connected_entities(
                            entity_name=plot['name'],
                            entity_type='plot_element',
                            session_id=session_id
                        )
                        if connected:
                            plot_connections.append({
                                'plot_name': plot['name'],
                                'connected_to': [
                                    {'name': entity['name'], 'type': entity['type']}
                                    for entity in connected
                                ]
                            })
                except Exception as e:
                    logger.warning(f"Could not retrieve plot connections: {e}")
                
                response_data = {
                    "status": "success",
                    "session_id": session_id,
                    "active_plots": [
                        {
                            "name": plot['name'],
                            "latest_mention": plot['content'][:200] + "...",
                            "last_updated": plot['timestamp'].isoformat(),
                            "attributes": plot['attributes']
                        }
                        for plot in sorted(active_plots, key=lambda x: x['timestamp'], reverse=True)[:15]
                    ],
                    "resolved_plots": [
                        {
                            "name": plot['name'],
                            "resolution": plot['content'][:200] + "...",
                            "resolved_at": plot['timestamp'].isoformat()
                        }
                        for plot in sorted(resolved_plots, key=lambda x: x['timestamp'], reverse=True)[:10]
                    ],
                    "plot_connections": plot_connections
                }
                
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(response_data, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error in get_plot_threads: {e}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ])
    
    def _register_get_context_stats_tool(self) -> None:
        """Register the get_context_stats tool"""
        
        @self.server.call_tool()
        async def get_context_stats(arguments: dict) -> CallToolResult:
            """Get context management statistics"""
            try:
                # Validate required arguments
                if 'session_id' not in arguments:
                    raise McpError("Missing required argument: session_id")
                
                session_id = arguments['session_id']
                
                # Get current context window
                context_window = await self.context_manager.build_context_window(session_id)
                pressure = self.context_manager.calculate_pressure(context_window)
                
                # Get memory counts from storage systems
                memory_stats = await self._get_memory_statistics(session_id)
                
                # Get relief history (if available)
                relief_history = getattr(self.context_manager.pressure_valve, 'relief_history', [])
                
                response_data = {
                    "status": "success",
                    "session_id": session_id,
                    "context_pressure": {
                        "current_pressure": pressure,
                        "pressure_percentage": f"{pressure:.1%}",
                        "threshold": self.config.context.pressure_threshold,
                        "needs_relief": pressure > self.config.context.pressure_threshold,
                        "tokens_used": context_window.total_tokens,
                        "tokens_limit": self.config.context.max_context_tokens
                    },
                    "context_window": {
                        "total_chunks": len(context_window.chunks),
                        "recent_chunks": len([c for c in context_window.chunks if c.chunk_type == 'recent']),
                        "memory_chunks": len([c for c in context_window.chunks if c.chunk_type == 'memory']),
                        "total_tokens": context_window.total_tokens
                    },
                    "memory_counts": memory_stats,
                    "relief_history": [
                        {
                            "timestamp": event.get('timestamp', 'unknown'),
                            "chunks_removed": event.get('chunks_removed', 0),
                            "tokens_freed": event.get('tokens_freed', 0),
                            "reason": event.get('reason', 'pressure_relief')
                        }
                        for event in relief_history[-10:]  # Last 10 relief events
                    ]
                }
                
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(response_data, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error in get_context_stats: {e}")
                error_response = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return CallToolResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_response, indent=2)
                    )
                ])
    
    def _register_resources(self) -> None:
        """Register MCP resources"""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources"""
            resources = [
                Resource(
                    uri="story://current-context",
                    name="Current Story Context",
                    description="Active context window for story writing",
                    mimeType="application/json"
                ),
                Resource(
                    uri="story://characters",
                    name="Story Characters",
                    description="All characters in current story session",
                    mimeType="application/json"
                ),
                Resource(
                    uri="story://memory-stats",
                    name="Memory Statistics",
                    description="Context management and memory statistics",
                    mimeType="application/json"
                )
            ]
            return ListResourcesResult(resources=resources)
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read specific resource content"""
            try:
                if uri == "story://current-context":
                    # Return current context for default session
                    session_id = "default"
                    context_window = await self.context_manager.build_context_window(session_id)
                    
                    content = {
                        "session_id": session_id,
                        "total_tokens": context_window.total_tokens,
                        "chunks": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                                "token_count": chunk.token_count,
                                "timestamp": chunk.timestamp.isoformat(),
                                "chunk_type": chunk.chunk_type
                            }
                            for chunk in context_window.chunks
                        ]
                    }
                    
                elif uri == "story://characters":
                    # Return character information for default session
                    session_id = "default"
                    memories = await self.context_manager.retrieval_system.get_relevant_memories(
                        query="character",
                        session_id=session_id,
                        max_results=100
                    )
                    
                    characters = {}
                    for memory in memories:
                        for entity in memory['chunk'].entities:
                            if entity.entity_type == 'character':
                                if entity.name not in characters:
                                    characters[entity.name] = {
                                        "name": entity.name,
                                        "mentions": 0,
                                        "attributes": set(),
                                        "last_mentioned": entity.metadata.get('last_mentioned', 'unknown')
                                    }
                                characters[entity.name]["mentions"] += 1
                                if entity.attributes:
                                    characters[entity.name]["attributes"].update(entity.attributes)
                    
                    # Convert sets to lists for JSON serialization
                    for char in characters.values():
                        char["attributes"] = list(char["attributes"])
                    
                    content = {
                        "session_id": session_id,
                        "character_count": len(characters),
                        "characters": list(characters.values())
                    }
                    
                elif uri == "story://memory-stats":
                    # Return memory statistics for default session
                    session_id = "default"
                    content = await self._get_memory_statistics(session_id)
                    
                else:
                    raise McpError(f"Unknown resource URI: {uri}")
                
                return ReadResourceResult([
                    TextContent(
                        type="text",
                        text=json.dumps(content, indent=2)
                    )
                ])
                
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                error_content = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                return ReadResourceResult([
                    TextContent(
                        type="text",
                        text=json.dumps(error_content, indent=2)
                    )
                ])
    
    async def _get_memory_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get memory statistics for a session"""
        try:
            # Get counts from SQLite store
            sqlite_stats = await self.context_manager.sqlite_store.get_session_stats(session_id)
            
            # Get counts from vector store
            vector_count = 0
            try:
                # This would need to be implemented in VectorStore
                vector_count = getattr(self.context_manager.vector_store, 'get_chunk_count', lambda x: 0)(session_id)
            except Exception:
                pass
            
            # Get counts from graph store
            graph_stats = {}
            try:
                graph_stats = await self.context_manager.graph_store.get_session_stats(session_id)
            except Exception:
                graph_stats = {"entities": 0, "relationships": 0}
            
            return {
                "sqlite_chunks": sqlite_stats.get("chunk_count", 0),
                "sqlite_sessions": sqlite_stats.get("session_count", 0),
                "vector_embeddings": vector_count,
                "graph_entities": graph_stats.get("entities", 0),
                "graph_relationships": graph_stats.get("relationships", 0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def shutdown(self) -> None:
        """Cleanup server resources"""
        if self.context_manager:
            await self.context_manager.shutdown()
        logger.info("Virtual Context MCP Server shutdown completed")


async def main():
    """Main entry point for running the MCP server"""
    server_instance = VirtualContextMCPServer()
    server = await server_instance.setup_server()
    
    # In a real implementation, this would start the MCP server
    logger.info("Virtual Context MCP Server ready")
    return server


if __name__ == "__main__":
    asyncio.run(main())