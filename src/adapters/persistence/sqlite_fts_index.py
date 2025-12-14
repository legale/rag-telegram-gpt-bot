"""SQLite FTS5 implementation of FTSIndex interface."""

from __future__ import annotations

import re
import unicodedata
import sqlite3
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, DatabaseError as SQLAlchemyDatabaseError

from src.core.interfaces import FTSIndex, ScoredDoc, SearchFilters
from src.storage.db import Database
from src.lib.syslog2 import *


class SqliteFTSIndex:
    """SQLite FTS5 adapter implementing FTSIndex protocol."""

    def __init__(self, database: Database):
        """
        Initialize the FTS5 index adapter.

        Args:
            database: Database instance from src.storage.db
        """
        self.db = database
        self._ensure_fts_tables()
        self._recovery_attempted = False  # Track if recovery was attempted to avoid loops

    def _ensure_fts_tables(self):
        """Ensure FTS5 virtual tables exist."""
        session = self.db.get_session()
        try:
            # Create messages_fts table if it doesn't exist
            session.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    msg_id UNINDEXED,
                    chat_id UNINDEXED,
                    from_id UNINDEXED,
                    ts UNINDEXED,
                    text,
                    content='messages',
                    content_rowid='rowid'
                )
            """))
            
            # Create chunks_fts table if it doesn't exist
            session.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED,
                    chat_id UNINDEXED,
                    text,
                    content='chunks',
                    content_rowid='rowid'
                )
            """))
            
            # Create triggers to keep FTS5 in sync with main tables
            # Messages triggers
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                    VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
                END
            """))
            
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE msg_id = old.msg_id;
                END
            """))
            
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                    DELETE FROM messages_fts WHERE msg_id = old.msg_id;
                    INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                    VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
                END
            """))
            
            # Chunks triggers
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(id, chat_id, text)
                    VALUES (new.id, new.chat_id, new.text);
                END
            """))
            
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE id = old.id;
                END
            """))
            
            session.execute(text("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE id = old.id;
                    INSERT INTO chunks_fts(id, chat_id, text)
                    VALUES (new.id, new.chat_id, new.text);
                END
            """))
            
            session.commit()
            
            # Populate FTS tables if they're empty (in case chunks were inserted before FTS table creation)
            self._populate_fts_if_empty(session)
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def _populate_fts_if_empty(self, session):
        """Populate FTS tables if they exist but are empty."""
        try:
            # Check if chunks_fts is empty
            result = session.execute(text("SELECT COUNT(*) FROM chunks_fts"))
            count = result.scalar()
            if count == 0:
                # Check if chunks table has data
                result = session.execute(text("SELECT COUNT(*) FROM chunks"))
                chunks_count = result.scalar()
                if chunks_count > 0:
                    syslog2(LOG_NOTICE, "populating empty chunks_fts table", chunks_count=chunks_count)
                    session.execute(text("""
                        INSERT INTO chunks_fts(id, chat_id, text)
                        SELECT id, chat_id, text FROM chunks
                    """))
                    session.commit()
            
            # Check if messages_fts is empty
            result = session.execute(text("SELECT COUNT(*) FROM messages_fts"))
            count = result.scalar()
            if count == 0:
                # Check if messages table has data
                result = session.execute(text("SELECT COUNT(*) FROM messages"))
                messages_count = result.scalar()
                if messages_count > 0:
                    syslog2(LOG_NOTICE, "populating empty messages_fts table", messages_count=messages_count)
                    session.execute(text("""
                        INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                        SELECT msg_id, chat_id, from_id, ts, text FROM messages
                    """))
                    session.commit()
        except Exception as e:
            # Don't fail if population fails - might be expected in some cases
            syslog2(LOG_DEBUG, "fts population check failed", error=str(e))

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for indexing/searching.

        Normalization includes:
        - Lowercase
        - ё -> е conversion
        - Punctuation removal

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        normalized = text.lower()
        
        # ё -> е
        normalized = normalized.replace('ё', 'е').replace('Ё', 'е')
        
        # Remove punctuation (keep spaces and alphanumeric)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize unicode (NFD -> NFC)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _check_database_integrity(self) -> bool:
        """
        Check database integrity using PRAGMA quick_check.

        Returns:
            True if database is healthy, False otherwise
        """
        session = self.db.get_session()
        try:
            # Get raw SQLite connection for PRAGMA commands
            raw_conn = session.connection().connection
            cursor = raw_conn.cursor()
            cursor.execute("PRAGMA quick_check")
            result = cursor.fetchone()
            return result and result[0] == 'ok'
        except Exception as e:
            syslog2(LOG_ERR, "database integrity check failed", error=str(e))
            return False
        finally:
            session.close()

    def _check_fts_table_integrity(self, table: str) -> bool:
        """
        Check if FTS table is accessible by attempting a simple query.

        Args:
            table: FTS table name ("chunks_fts" or "messages_fts")

        Returns:
            True if table is accessible, False otherwise
        """
        session = self.db.get_session()
        try:
            # Try a simple query to check if table is accessible
            session.execute(text(f"SELECT COUNT(*) FROM {table} LIMIT 1"))
            session.commit()
            return True
        except (sqlite3.DatabaseError, Exception) as e:
            syslog2(LOG_WARNING, "fts table integrity check failed", table=table, error=str(e))
            return False
        finally:
            session.close()

    def _rebuild_fts_table(self, table: str) -> bool:
        """
        Rebuild a corrupted FTS table.

        Args:
            table: FTS table name ("chunks_fts" or "messages_fts")

        Returns:
            True if rebuild was successful, False otherwise
        """
        session = self.db.get_session()
        try:
            syslog2(LOG_WARNING, "rebuilding fts table", table=table)
            
            # Drop corrupted table
            session.execute(text(f"DROP TABLE IF EXISTS {table}"))
            
            # Recreate table
            if table == "chunks_fts":
                session.execute(text("""
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        id UNINDEXED,
                        chat_id UNINDEXED,
                        text,
                        content='chunks',
                        content_rowid='rowid'
                    )
                """))
                
                # Recreate triggers
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(id, chat_id, text)
                        VALUES (new.id, new.chat_id, new.text);
                    END
                """))
                
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE id = old.id;
                    END
                """))
                
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE id = old.id;
                        INSERT INTO chunks_fts(id, chat_id, text)
                        VALUES (new.id, new.chat_id, new.text);
                    END
                """))
                
                # Rebuild from chunks table
                session.execute(text("""
                    INSERT INTO chunks_fts(id, chat_id, text)
                    SELECT id, chat_id, text FROM chunks
                """))
                
            elif table == "messages_fts":
                session.execute(text("""
                    CREATE VIRTUAL TABLE messages_fts USING fts5(
                        msg_id UNINDEXED,
                        chat_id UNINDEXED,
                        from_id UNINDEXED,
                        ts UNINDEXED,
                        text,
                        content='messages',
                        content_rowid='rowid'
                    )
                """))
                
                # Recreate triggers
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                        VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
                    END
                """))
                
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                        DELETE FROM messages_fts WHERE msg_id = old.msg_id;
                    END
                """))
                
                session.execute(text("""
                    CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                        DELETE FROM messages_fts WHERE msg_id = old.msg_id;
                        INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                        VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
                    END
                """))
                
                # Rebuild from messages table
                session.execute(text("""
                    INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                    SELECT msg_id, chat_id, from_id, ts, text FROM messages
                """))
            
            session.commit()
            syslog2(LOG_NOTICE, "fts table rebuilt successfully", table=table)
            return True
            
        except Exception as e:
            session.rollback()
            syslog2(LOG_ERR, "fts table rebuild failed", table=table, error=str(e))
            return False
        finally:
            session.close()

    def _recover_fts_tables(self, table: str) -> bool:
        """
        Attempt to recover FTS tables if corrupted.

        Args:
            table: FTS table name ("chunks_fts" or "messages_fts")

        Returns:
            True if recovery was successful, False otherwise
        """
        if self._recovery_attempted:
            # Avoid infinite recovery loops
            syslog2(LOG_WARNING, "fts recovery already attempted, skipping", table=table)
            return False
        
        self._recovery_attempted = True
        
        # Check database integrity first
        if not self._check_database_integrity():
            syslog2(LOG_ERR, "database integrity check failed, cannot recover fts table", table=table)
            return False
        
        # Attempt to rebuild the table
        return self._rebuild_fts_table(table)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None,
        table: str = "chunks_fts"  # "chunks_fts" or "messages_fts"
    ) -> List[ScoredDoc]:
        """
        Search using FTS5.

        Args:
            query: Search query text (will be normalized)
            top_k: Number of results
            filters: Optional filters (author, time_range, chat_id, etc.)
            table: Which FTS table to search ("chunks_fts" or "messages_fts")

        Returns:
            List of ScoredDoc with document IDs and scores
        """
        if not query:
            return []

        # Normalize query
        normalized_query = self.normalize_text(query)
        if not normalized_query:
            return []

        # Build FTS5 query (escape special characters)
        # FTS5 uses double quotes for phrases, but we'll use simple term matching
        fts_query = normalized_query.replace('"', '""')
        
        session = self.db.get_session()
        try:
            # Build WHERE clause for filters
            where_clauses = []
            params = {}
            
            if filters:
                if filters.chat_id:
                    where_clauses.append(f"{table}.chat_id = :chat_id")
                    params["chat_id"] = filters.chat_id
                
                if filters.author:
                    if table == "messages_fts":
                        where_clauses.append(f"{table}.from_id = :author")
                    else:
                        # For chunks, we need to join with messages or check metadata
                        # For now, we'll skip author filter for chunks
                        pass
                    params["author"] = filters.author
                
                if filters.time_from:
                    if table == "messages_fts":
                        where_clauses.append(f"{table}.ts >= :time_from")
                    else:
                        where_clauses.append(f"chunks.ts_from >= :time_from")
                    params["time_from"] = filters.time_from
                
                if filters.time_to:
                    if table == "messages_fts":
                        where_clauses.append(f"{table}.ts <= :time_to")
                    else:
                        where_clauses.append(f"chunks.ts_to <= :time_to")
                    params["time_to"] = filters.time_to
            
            # Build WHERE clause combining filters and FTS5 match
            where_parts = [f"{table} MATCH :query"]
            join_sql = ""
            
            # For chunks, we need to join with main table for time filters
            if table == "chunks_fts" and filters and (filters.time_from or filters.time_to):
                # Need to join with chunks table for time filters
                join_sql = "JOIN chunks ON chunks_fts.id = chunks.id"
                if filters.time_from:
                    where_parts.append("chunks.ts_from >= :time_from")
                if filters.time_to:
                    where_parts.append("chunks.ts_to <= :time_to")
            elif where_clauses:
                where_parts.extend(where_clauses)
            
            where_sql = "WHERE " + " AND ".join(where_parts)
            
            # FTS5 search query
            # bm25() gives better ranking than simple rank
            if join_sql:
                # Use JOIN for chunks with time filters
                sql = f"""
                    SELECT 
                        {table}.rowid,
                        {table}.id as doc_id,
                        bm25({table}) as score
                    FROM {table}
                    {join_sql}
                    {where_sql}
                    ORDER BY bm25({table})
                    LIMIT :top_k
                """
            else:
                sql = f"""
                    SELECT 
                        {table}.rowid,
                        {table}.{'msg_id' if table == 'messages_fts' else 'id'} as doc_id,
                        bm25({table}) as score
                    FROM {table}
                    {where_sql}
                    ORDER BY bm25({table})
                    LIMIT :top_k
                """
            
            params["query"] = fts_query
            params["top_k"] = top_k
            
            result = session.execute(text(sql), params)
            rows = result.fetchall()
            
            scored_docs = []
            for row in rows:
                doc_id = row[1]  # doc_id column
                score = row[2]   # score column
                # FTS5 bm25 returns negative scores (lower is better), convert to positive
                # Higher score = better match
                normalized_score = abs(score) if score < 0 else score
                
                scored_docs.append(ScoredDoc(
                    id=str(doc_id),
                    score=normalized_score,
                    meta={}
                ))
            
            return scored_docs
            
        except (sqlite3.DatabaseError, OperationalError, SQLAlchemyDatabaseError) as e:
            syslog2(LOG_ERR, "database error during fts search", table=table, error=str(e))
            # Attempt recovery if not already attempted
            if not self._recovery_attempted:
                syslog2(LOG_WARNING, "attempting fts table recovery after database error", table=table)
                session.close()  # Close before recovery
                if self._recover_fts_tables(table):
                    self._recovery_attempted = False  # Reset flag
                    # Retry search after recovery
                    return self.search(query, top_k, filters, table)
            return []
        except Exception as e:
            # Check if it's a database-related error wrapped in generic Exception
            error_str = str(e)
            if "database disk image is malformed" in error_str.lower() or "database" in error_str.lower():
                syslog2(LOG_ERR, "database error during fts search (wrapped)", table=table, error=error_str)
                # Attempt recovery if not already attempted
                if not self._recovery_attempted:
                    syslog2(LOG_WARNING, "attempting fts table recovery after database error", table=table)
                    session.close()  # Close before recovery
                    if self._recover_fts_tables(table):
                        self._recovery_attempted = False  # Reset flag
                        # Retry search after recovery
                        return self.search(query, top_k, filters, table)
            else:
                syslog2(LOG_ERR, "unexpected error during fts search", table=table, error=error_str)
            return []
        finally:
            session.close()

    def search_messages(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None
    ) -> List[ScoredDoc]:
        """Search in messages_fts table."""
        return self.search(query, top_k, filters, table="messages_fts")

    def search_chunks(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None
    ) -> List[ScoredDoc]:
        """Search in chunks_fts table."""
        return self.search(query, top_k, filters, table="chunks_fts")

