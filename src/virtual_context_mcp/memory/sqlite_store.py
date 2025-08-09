"""SQLite storage implementation for conversation chunks."""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..chunking.chunk import ContextChunk


class SQLiteStore:
    """SQLite-based storage for conversation chunks and sessions."""
    
    def __init__(self, db_path: str):
        """Initialize SQLite database with schema."""
        self.db_path = db_path
        self._lock = asyncio.Lock()
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the storage system."""
        await self.init_database()
    
    async def init_database(self) -> None:
        """Create tables if they don't exist."""
        async with self._lock:
            await asyncio.to_thread(self._create_schema)
    
    def _create_schema(self) -> None:
        """Create database schema in synchronous context."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    chunk_type TEXT DEFAULT 'conversation',
                    entities_json TEXT,
                    embedding_json TEXT
                )
            """)
            
            # Create indexes for chunks
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp)")
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def store_chunk(self, chunk: ContextChunk) -> None:
        """Store single chunk."""
        await self.store_chunks([chunk])
    
    async def store_chunks(self, chunks: List[ContextChunk]) -> None:
        """Store multiple chunks in transaction."""
        if not chunks:
            return
        
        async with self._lock:
            await asyncio.to_thread(self._store_chunks_sync, chunks)
    
    def _store_chunks_sync(self, chunks: List[ContextChunk]) -> None:
        """Store chunks in synchronous context with transaction."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            for chunk in chunks:
                # Prepare data for storage
                entities_json = json.dumps(chunk.entities) if chunk.entities else None
                embedding_json = json.dumps(chunk.embedding) if chunk.embedding else None
                timestamp_str = chunk.timestamp.isoformat()
                
                # Insert or replace chunk
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, session_id, content, token_count, timestamp, chunk_type, entities_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.session_id,
                    chunk.content,
                    chunk.token_count,
                    timestamp_str,
                    chunk.chunk_type,
                    entities_json,
                    embedding_json
                ))
                
                # Update session last_active
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, last_active)
                    VALUES (?, ?)
                """, (chunk.session_id, timestamp_str))
            
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def get_chunks_by_session(self, session_id: str, limit: Optional[int] = None) -> List[ContextChunk]:
        """Retrieve chunks for session, most recent first."""
        async with self._lock:
            return await asyncio.to_thread(self._get_chunks_by_session_sync, session_id, limit)
    
    def _get_chunks_by_session_sync(self, session_id: str, limit: Optional[int] = None) -> List[ContextChunk]:
        """Retrieve chunks in synchronous context."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            query = """
                SELECT id, session_id, content, token_count, timestamp, chunk_type, entities_json, embedding_json
                FROM chunks 
                WHERE session_id = ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                (chunk_id, session_id, content, token_count, timestamp_str, 
                 chunk_type, entities_json, embedding_json) = row
                
                # Parse JSON fields
                entities = json.loads(entities_json) if entities_json else None
                embedding = json.loads(embedding_json) if embedding_json else None
                timestamp = datetime.fromisoformat(timestamp_str)
                
                chunk = ContextChunk(
                    id=chunk_id,
                    session_id=session_id,
                    content=content,
                    token_count=token_count,
                    timestamp=timestamp,
                    chunk_type=chunk_type,
                    entities=entities,
                    embedding=embedding
                )
                chunks.append(chunk)
            
            return chunks
        finally:
            conn.close()
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[ContextChunk]:
        """Retrieve specific chunk."""
        async with self._lock:
            return await asyncio.to_thread(self._get_chunk_by_id_sync, chunk_id)
    
    def _get_chunk_by_id_sync(self, chunk_id: str) -> Optional[ContextChunk]:
        """Retrieve chunk in synchronous context."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, session_id, content, token_count, timestamp, chunk_type, entities_json, embedding_json
                FROM chunks 
                WHERE id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            (chunk_id, session_id, content, token_count, timestamp_str, 
             chunk_type, entities_json, embedding_json) = row
            
            # Parse JSON fields
            entities = json.loads(entities_json) if entities_json else None
            embedding = json.loads(embedding_json) if embedding_json else None
            timestamp = datetime.fromisoformat(timestamp_str)
            
            return ContextChunk(
                id=chunk_id,
                session_id=session_id,
                content=content,
                token_count=token_count,
                timestamp=timestamp,
                chunk_type=chunk_type,
                entities=entities,
                embedding=embedding
            )
        finally:
            conn.close()
    
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete chunks and return count deleted."""
        if not chunk_ids:
            return 0
        
        async with self._lock:
            return await asyncio.to_thread(self._delete_chunks_sync, chunk_ids)
    
    def _delete_chunks_sync(self, chunk_ids: List[str]) -> int:
        """Delete chunks in synchronous context."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create placeholders for IN clause
            placeholders = ','.join('?' * len(chunk_ids))
            
            cursor.execute(f"""
                DELETE FROM chunks 
                WHERE id IN ({placeholders})
            """, chunk_ids)
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    async def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get session information."""
        async with self._lock:
            return await asyncio.to_thread(self._get_session_info_sync, session_id)
    
    def _get_session_info_sync(self, session_id: str) -> Optional[dict]:
        """Get session info in synchronous context."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, created_at, last_active
                FROM sessions 
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            session_id, created_at, last_active = row
            return {
                "session_id": session_id,
                "created_at": created_at,
                "last_active": last_active
            }
        finally:
            conn.close()
    
    async def close(self) -> None:
        """Close database connections (no-op for SQLite)."""
        pass