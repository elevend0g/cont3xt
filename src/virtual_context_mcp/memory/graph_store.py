"""
Neo4j knowledge graph storage for story entities and relationships.

This module provides functionality to store and query story entities and their
relationships using Neo4j as a knowledge graph database, enabling complex
relationship traversals and entity connection discovery.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from ..entities.story_entities import StoryEntity

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j-based knowledge graph storage for story entities and relationships."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            await self._verify_connectivity()
    
    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
    
    async def _verify_connectivity(self) -> None:
        """Verify that we can connect to the Neo4j database."""
        if not self._driver:
            raise RuntimeError("Driver not initialized")
        
        async with self._driver.session() as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            if not record or record["test"] != 1:
                raise RuntimeError("Failed to verify Neo4j connectivity")
    
    async def initialize(self) -> None:
        """Initialize the graph store."""
        await self.connect()
        await self.init_schema()
    
    async def init_schema(self) -> None:
        """Create constraints and indexes for optimal performance."""
        if not self._driver:
            await self.connect()
        
        schema_queries = [
            # Character constraints
            "CREATE CONSTRAINT character_name_session IF NOT EXISTS "
            "FOR (c:Character) REQUIRE (c.name, c.session_id) IS UNIQUE",
            
            # Location constraints  
            "CREATE CONSTRAINT location_name_session IF NOT EXISTS "
            "FOR (l:Location) REQUIRE (l.name, l.session_id) IS UNIQUE",
            
            # Plot element constraints
            "CREATE CONSTRAINT plot_name_session IF NOT EXISTS "
            "FOR (p:PlotElement) REQUIRE (p.name, p.session_id) IS UNIQUE",
            
            # Theme constraints
            "CREATE CONSTRAINT theme_name_session IF NOT EXISTS "
            "FOR (t:Theme) REQUIRE (t.name, t.session_id) IS UNIQUE",
            
            # Chunk constraints
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            
            # Session constraints
            "CREATE CONSTRAINT session_id IF NOT EXISTS "
            "FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX character_session_idx IF NOT EXISTS "
            "FOR (c:Character) ON (c.session_id)",
            
            "CREATE INDEX location_session_idx IF NOT EXISTS "
            "FOR (l:Location) ON (l.session_id)",
            
            "CREATE INDEX entity_confidence_idx IF NOT EXISTS "
            "FOR (n) ON (n.confidence) WHERE n:Character OR n:Location OR n:PlotElement OR n:Theme"
        ]
        
        async with self._driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.debug(f"Executed schema query: {query}")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {query} - {e}")
        
        self._initialized = True
        logger.info("Neo4j schema initialization completed")
    
    async def store_entities(self, entities: Dict[str, List[StoryEntity]], chunk_id: str, session_id: str) -> None:
        """Store entities and their relationships from a chunk."""
        if not self._initialized:
            await self.init_schema()
        
        # Create session and chunk nodes
        await self._ensure_session_exists(session_id)
        await self._ensure_chunk_exists(chunk_id, session_id)
        
        # Store all entities
        stored_entities = set()
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity_type == "character":
                    await self.create_character(entity.name, session_id, entity.attributes)
                elif entity_type == "location":
                    await self.create_location(entity.name, session_id, entity.attributes)
                elif entity_type == "plot_element":
                    await self.create_plot_element(entity.name, session_id, entity.attributes)
                elif entity_type == "theme":
                    await self.create_theme(entity.name, session_id, entity.attributes)
                
                # Link entity to chunk
                await self._link_entity_to_chunk(entity.name, entity_type, chunk_id, session_id)
                stored_entities.add((entity.name, entity_type))
        
        # Create relationships between entities based on co-occurrence
        await self._create_cooccurrence_relationships(stored_entities, chunk_id, session_id)
        
        logger.debug(f"Stored {sum(len(el) for el in entities.values())} entities for chunk {chunk_id}")
    
    async def create_character(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update character node."""
        if not self._driver:
            await self.connect()
        
        query = """
        MERGE (c:Character {name: $name, session_id: $session_id})
        SET c += $attributes,
            c.updated_at = datetime(),
            c.entity_type = 'character'
        RETURN c
        """
        
        attrs = attributes or {}
        attrs.update({
            "name": name,
            "session_id": session_id,
            "created_at": attrs.get("created_at", "datetime()")
        })
        
        async with self._driver.session() as session:
            await session.run(query, name=name, session_id=session_id, attributes=attrs)
    
    async def create_location(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update location node."""
        if not self._driver:
            await self.connect()
        
        query = """
        MERGE (l:Location {name: $name, session_id: $session_id})
        SET l += $attributes,
            l.updated_at = datetime(),
            l.entity_type = 'location'
        RETURN l
        """
        
        attrs = attributes or {}
        attrs.update({
            "name": name,
            "session_id": session_id,
            "created_at": attrs.get("created_at", "datetime()")
        })
        
        async with self._driver.session() as session:
            await session.run(query, name=name, session_id=session_id, attributes=attrs)
    
    async def create_plot_element(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update plot element node."""
        if not self._driver:
            await self.connect()
        
        query = """
        MERGE (p:PlotElement {name: $name, session_id: $session_id})
        SET p += $attributes,
            p.updated_at = datetime(),
            p.entity_type = 'plot_element'
        RETURN p
        """
        
        attrs = attributes or {}
        attrs.update({
            "name": name,
            "session_id": session_id,
            "created_at": attrs.get("created_at", "datetime()")
        })
        
        async with self._driver.session() as session:
            await session.run(query, name=name, session_id=session_id, attributes=attrs)
    
    async def create_theme(self, name: str, session_id: str, attributes: Dict[str, Any] = None) -> None:
        """Create or update theme node."""
        if not self._driver:
            await self.connect()
        
        query = """
        MERGE (t:Theme {name: $name, session_id: $session_id})
        SET t += $attributes,
            t.updated_at = datetime(),
            t.entity_type = 'theme'
        RETURN t
        """
        
        attrs = attributes or {}
        attrs.update({
            "name": name,
            "session_id": session_id,
            "created_at": attrs.get("created_at", "datetime()")
        })
        
        async with self._driver.session() as session:
            await session.run(query, name=name, session_id=session_id, attributes=attrs)
    
    async def create_relationship(self, entity1: str, relationship: str, entity2: str, session_id: str, 
                                attributes: Dict[str, Any] = None) -> None:
        """Create relationship between entities."""
        if not self._driver:
            await self.connect()
        
        # Use parameterized query to prevent injection
        query = f"""
        MATCH (a {{name: $entity1, session_id: $session_id}})
        MATCH (b {{name: $entity2, session_id: $session_id}})
        MERGE (a)-[r:`{relationship}`]->(b)
        SET r += $attributes,
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        RETURN r
        """
        
        attrs = attributes or {}
        
        async with self._driver.session() as session:
            await session.run(
                query, 
                entity1=entity1, 
                entity2=entity2, 
                session_id=session_id,
                attributes=attrs
            )
    
    async def get_character_relationships(self, character_name: str, session_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a character."""
        if not self._driver:
            await self.connect()
        
        query = """
        MATCH (c:Character {name: $character_name, session_id: $session_id})
        MATCH (c)-[r]-(other)
        RETURN 
            other.name as connected_entity,
            labels(other) as entity_types,
            type(r) as relationship_type,
            startNode(r).name as source,
            endNode(r).name as target,
            r as relationship_data
        ORDER BY relationship_type, connected_entity
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, character_name=character_name, session_id=session_id)
            relationships = []
            async for record in result:
                relationships.append({
                    "connected_entity": record["connected_entity"],
                    "entity_types": record["entity_types"],
                    "relationship_type": record["relationship_type"],
                    "source": record["source"],
                    "target": record["target"],
                    "relationship_data": dict(record["relationship_data"])
                })
            return relationships
    
    async def get_connected_entities(self, entity_names: List[str], session_id: str, max_depth: int = 2) -> List[str]:
        """Get entity names connected to any of the input entities within max_depth levels."""
        if not entity_names or not self._driver:
            return []
        
        if not self._driver:
            await self.connect()
        
        # Build query to find entities connected to any of the input entities
        query = f"""
        MATCH (start {{session_id: $session_id}})
        WHERE start.name IN $entity_names
        MATCH path = (start)-[*1..{max_depth}]-(connected)
        WHERE connected.session_id = $session_id
        AND NOT connected.name IN $entity_names
        RETURN DISTINCT connected.name as name
        ORDER BY name
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, entity_names=entity_names, session_id=session_id)
            connected_names = []
            async for record in result:
                connected_names.append(record["name"])
            return connected_names
    
    async def get_connected_entities_detailed(self, entity_name: str, session_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities connected within depth levels with detailed information."""
        if not self._driver:
            await self.connect()
        
        query = f"""
        MATCH (start {{name: $entity_name, session_id: $session_id}})
        MATCH path = (start)-[*1..{depth}]-(connected)
        WHERE connected.session_id = $session_id
        RETURN DISTINCT 
            connected.name as name,
            labels(connected) as entity_types,
            connected.entity_type as entity_type,
            length(path) as distance,
            connected as entity_data
        ORDER BY distance, name
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, entity_name=entity_name, session_id=session_id)
            entities = []
            async for record in result:
                entities.append({
                    "name": record["name"],
                    "entity_types": record["entity_types"],
                    "entity_type": record["entity_type"],
                    "distance": record["distance"],
                    "entity_data": dict(record["entity_data"])
                })
            return entities
    
    async def are_entities_connected(self, entity1_name: str, entity2_name: str, session_id: str, max_depth: int = 3) -> bool:
        """Check if two entities are connected within max_depth relationships."""
        if not self._driver:
            await self.connect()
        
        query = f"""
        MATCH (entity1 {{name: $entity1_name, session_id: $session_id}})
        MATCH (entity2 {{name: $entity2_name, session_id: $session_id}})
        MATCH path = shortestPath((entity1)-[*1..{max_depth}]-(entity2))
        RETURN path IS NOT NULL as connected
        """
        
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    query, 
                    entity1_name=entity1_name, 
                    entity2_name=entity2_name, 
                    session_id=session_id
                )
                record = await result.single()
                return record["connected"] if record else False
        except Exception:
            return False
    
    async def search_by_relationship(self, relationship_type: str, session_id: str) -> List[Dict[str, Any]]:
        """Find entities by relationship type."""
        if not self._driver:
            await self.connect()
        
        # Use parameterized query for the relationship type
        query = f"""
        MATCH (a)-[r:`{relationship_type}`]->(b)
        WHERE a.session_id = $session_id AND b.session_id = $session_id
        RETURN 
            a.name as source_entity,
            labels(a) as source_types,
            b.name as target_entity,
            labels(b) as target_types,
            r as relationship_data
        ORDER BY source_entity, target_entity
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, session_id=session_id)
            relationships = []
            async for record in result:
                relationships.append({
                    "source_entity": record["source_entity"],
                    "source_types": record["source_types"],
                    "target_entity": record["target_entity"],
                    "target_types": record["target_types"],
                    "relationship_data": dict(record["relationship_data"])
                })
            return relationships
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session."""
        if not self._driver:
            await self.connect()
        
        query = """
        MATCH (n {session_id: $session_id})
        RETURN 
            labels(n) as node_labels,
            count(n) as count
        ORDER BY count DESC
        """
        
        async with self._driver.session() as session:
            result = await session.run(query, session_id=session_id)
            summary = {"session_id": session_id, "entity_counts": {}}
            async for record in result:
                labels = record["node_labels"]
                count = record["count"]
                for label in labels:
                    if label not in ["Entity"]:  # Skip generic labels
                        summary["entity_counts"][label] = summary["entity_counts"].get(label, 0) + count
            return summary
    
    async def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure session node exists."""
        query = """
        MERGE (s:Session {session_id: $session_id})
        SET s.created_at = coalesce(s.created_at, datetime()),
            s.updated_at = datetime()
        RETURN s
        """
        
        async with self._driver.session() as session:
            await session.run(query, session_id=session_id)
    
    async def _ensure_chunk_exists(self, chunk_id: str, session_id: str) -> None:
        """Ensure chunk node exists and is linked to session."""
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.session_id = $session_id,
            c.created_at = coalesce(c.created_at, datetime()),
            c.updated_at = datetime()
        WITH c
        MATCH (s:Session {session_id: $session_id})
        MERGE (s)-[:CONTAINS]->(c)
        RETURN c
        """
        
        async with self._driver.session() as session:
            await session.run(query, chunk_id=chunk_id, session_id=session_id)
    
    async def _link_entity_to_chunk(self, entity_name: str, entity_type: str, chunk_id: str, session_id: str) -> None:
        """Link entity to the chunk where it was found."""
        # Map entity types to node labels
        label_map = {
            "character": "Character",
            "location": "Location", 
            "plot_element": "PlotElement",
            "theme": "Theme"
        }
        
        label = label_map.get(entity_type, "Entity")
        
        query = f"""
        MATCH (e:`{label}` {{name: $entity_name, session_id: $session_id}})
        MATCH (c:Chunk {{chunk_id: $chunk_id}})
        MERGE (c)-[:MENTIONS]->(e)
        """
        
        async with self._driver.session() as session:
            await session.run(query, entity_name=entity_name, chunk_id=chunk_id, session_id=session_id)
    
    async def _create_cooccurrence_relationships(self, entities: Set[tuple], chunk_id: str, session_id: str) -> None:
        """Create CO_OCCURS relationships between entities found in the same chunk."""
        entities_list = list(entities)
        
        for i, (entity1_name, entity1_type) in enumerate(entities_list):
            for entity2_name, entity2_type in entities_list[i+1:]:
                # Create bidirectional co-occurrence relationships
                await self._create_cooccurrence_relationship(
                    entity1_name, entity1_type, entity2_name, entity2_type, chunk_id, session_id
                )
    
    async def _create_cooccurrence_relationship(self, entity1_name: str, entity1_type: str, 
                                              entity2_name: str, entity2_type: str,
                                              chunk_id: str, session_id: str) -> None:
        """Create a CO_OCCURS relationship between two entities."""
        label_map = {
            "character": "Character",
            "location": "Location", 
            "plot_element": "PlotElement",
            "theme": "Theme"
        }
        
        label1 = label_map.get(entity1_type, "Entity")
        label2 = label_map.get(entity2_type, "Entity")
        
        query = f"""
        MATCH (e1:`{label1}` {{name: $entity1_name, session_id: $session_id}})
        MATCH (e2:`{label2}` {{name: $entity2_name, session_id: $session_id}})
        MERGE (e1)-[r:CO_OCCURS]-(e2)
        SET r.chunk_id = $chunk_id,
            r.co_occurrence_count = coalesce(r.co_occurrence_count, 0) + 1,
            r.updated_at = datetime()
        """
        
        async with self._driver.session() as session:
            await session.run(
                query, 
                entity1_name=entity1_name, 
                entity2_name=entity2_name,
                chunk_id=chunk_id,
                session_id=session_id
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()