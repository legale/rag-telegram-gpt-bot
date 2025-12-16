"""SQLite implementations for storage interfaces."""

from __future__ import annotations

import re
import unicodedata
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, DatabaseError as SQLAlchemyDatabaseError

from src.core.domain import Message, Chunk, TopicUpdate
from src.core.interfaces import MessageStore, ChunkStore, FTSIndex, ScoredDoc, SearchFilters
from src.storage.db import Database, MessageModel, MessageMetaModel, ChunkModel
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
        
        # Apply normalization steps
        normalized = self._to_lowercase(text)
        normalized = self._replace_yo(normalized)
        normalized = self._remove_punctuation(normalized)
        
        # Normalize unicode (NFD -> NFC)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _to_lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Text to convert
            
        Returns:
            Lowercase text
        """
        return text.lower()
    
    def _replace_yo(self, text: str) -> str:
        """
        Replace ё and Ё with е.
        
        Args:
            text: Text to process
            
        Returns:
            Text with ё replaced by е
        """
        return text.replace('ё', 'е').replace('Ё', 'е')
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text, keeping spaces and alphanumeric characters.
        
        Args:
            text: Text to process
            
        Returns:
            Text with punctuation removed
        """
        return re.sub(r'[^\w\s]', ' ', text)

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

    def _normalize_query_text(self, query: str) -> str:
        """
        Normalize query text for FTS5 search.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized and escaped FTS5 query string, or empty string if query is invalid
        """
        if not query:
            return ""
        
        # Normalize query
        normalized_query = self.normalize_text(query)
        if not normalized_query:
            return ""
        
        # Build FTS5 query (escape special characters)
        # FTS5 uses double quotes for phrases, but we'll use simple term matching
        return normalized_query.replace('"', '""')
    
    def _build_where_clause(
        self,
        table: str,
        filters: Optional[SearchFilters],
        fts_query: str
    ) -> Tuple[str, str, Dict]:
        """
        Build WHERE clause and JOIN SQL for FTS search.
        
        Args:
            table: FTS table name ("chunks_fts" or "messages_fts")
            filters: Optional search filters
            fts_query: Normalized FTS query string
            
        Returns:
            Tuple of (where_sql, join_sql, params_dict)
        """
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
        params["query"] = fts_query
        
        return where_sql, join_sql, params
    
    def _execute_fts_query(
        self,
        table: str,
        where_sql: str,
        join_sql: str,
        params: Dict,
        top_k: int
    ) -> List[ScoredDoc]:
        """
        Execute FTS5 search query and return results.
        
        Args:
            table: FTS table name ("chunks_fts" or "messages_fts")
            where_sql: WHERE clause SQL
            join_sql: JOIN clause SQL (may be empty)
            params: Query parameters dictionary
            top_k: Number of results to return
            
        Returns:
            List of ScoredDoc with document IDs and scores
        """
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
        
        params["top_k"] = top_k
        
        session = self.db.get_session()
        try:
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
        finally:
            session.close()
    
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
        # Normalize query text
        fts_query = self._normalize_query_text(query)
        if not fts_query:
            return []
        
        # Build WHERE clause and JOIN SQL
        where_sql, join_sql, params = self._build_where_clause(table, filters, fts_query)
        
        try:
            # Execute FTS query
            return self._execute_fts_query(table, where_sql, join_sql, params, top_k)
            
        except (sqlite3.DatabaseError, OperationalError, SQLAlchemyDatabaseError) as e:
            syslog2(LOG_ERR, "database error during fts search", table=table, error=str(e))
            # Attempt recovery if not already attempted
            if not self._recovery_attempted:
                syslog2(LOG_WARNING, "attempting fts table recovery after database error", table=table)
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
                    if self._recover_fts_tables(table):
                        self._recovery_attempted = False  # Reset flag
                        # Retry search after recovery
                        return self.search(query, top_k, filters, table)
            else:
                syslog2(LOG_ERR, "unexpected error during fts search", table=table, error=error_str)
            return []

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


class SqliteMessageStore:
    """SQLite adapter implementing MessageStore protocol."""

    def __init__(self, database: Database):
        """
        Initialize the adapter with a Database instance.

        Args:
            database: Database instance from src.storage.db
        """
        self.db = database
        self._fts_index: Optional[SqliteFTSIndex] = None

    def _convert_messages_to_dicts(self, messages: List[Message]) -> List[Dict]:
        """
        Convert domain Messages to dict format expected by Database.add_messages_batch.
        
        Args:
            messages: List of Message domain objects
            
        Returns:
            List of message dictionaries
        """
        message_dicts = []
        for msg in messages:
            message_dicts.append({
                "msg_id": msg.id,
                "chat_id": msg.chat_id,
                "ts": msg.timestamp,
                "from_id": msg.from_id,
                "text": msg.text,
            })
        return message_dicts

    def _save_message_metadata(self, messages: List[Message]) -> None:
        """
        Save message metadata to database.
        
        Args:
            messages: List of Message domain objects with metadata
        """
        session = self.db.get_session()
        try:
            for msg in messages:
                if msg.meta:
                    # Check if meta already exists
                    existing_meta = session.query(MessageMetaModel).filter(
                        MessageMetaModel.msg_id == msg.id
                    ).first()
                    
                    if existing_meta:
                        # Update existing metadata
                        existing_meta.meta_json = json.dumps(msg.meta)
                    else:
                        # Create new metadata entry
                        meta_model = MessageMetaModel(
                            msg_id=msg.id,
                            meta_json=json.dumps(msg.meta)
                        )
                        session.add(meta_model)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_batch(self, messages: List[Message]) -> int:
        """
        Save a batch of messages to the database.

        Args:
            messages: List of Message domain objects

        Returns:
            Number of messages actually saved (excluding duplicates)
        """
        if not messages:
            return 0

        # Convert domain Messages to dict format
        message_dicts = self._convert_messages_to_dicts(messages)

        # Save messages
        saved_count = self.db.add_messages_batch(message_dicts)

        # Save metadata separately if present
        self._save_message_metadata(messages)

        # FTS5 indexes are updated automatically via triggers
        # But we ensure FTS tables exist
        if self._fts_index is None:
            self._fts_index = SqliteFTSIndex(self.db)
        
        return saved_count

    def get_fts_index(self) -> FTSIndex:
        """
        Get FTS5 index for messages.

        Returns:
            FTSIndex instance for searching messages
        """
        if self._fts_index is None:
            self._fts_index = SqliteFTSIndex(self.db)
        return self._fts_index

    def get_by_chat(self, chat_id: str, limit: int, offset: int) -> List[Message]:
        """
        Get messages for a specific chat with pagination.

        Args:
            chat_id: Chat ID to filter by
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of Message domain objects
        """
        # Fetch messages from database
        models, session = self._fetch_messages_from_db(chat_id, limit, offset)
        
        # Convert to domain objects
        return self._convert_to_domain_messages(models, session)
    
    def _fetch_messages_from_db(self, chat_id: str, limit: int, offset: int) -> tuple[List[MessageModel], object]:
        """
        Fetch message models from database by chat ID.
        
        Args:
            chat_id: Chat ID to filter by
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            Tuple of (list of MessageModel instances, database session)
        """
        session = self.db.get_session()
        try:
            # Query with limit and offset
            query = session.query(MessageModel).filter(
                MessageModel.chat_id == chat_id
            ).order_by(MessageModel.ts).offset(offset).limit(limit)
            
            models = query.all()
            return models, session
        except Exception:
            session.close()
            raise
    
    def _convert_to_domain_messages(self, models: List[MessageModel], session) -> List[Message]:
        """
        Convert MessageModel instances to domain Message objects.
        
        Args:
            models: List of MessageModel instances
            session: Database session for loading metadata
            
        Returns:
            List of Message domain objects
        """
        try:
            return [self._model_to_domain(model, session) for model in models]
        finally:
            session.close()

    def _calculate_time_window(self, time_point: datetime, window_sec: int) -> tuple[datetime, datetime]:
        """
        Calculate time window boundaries.
        
        Args:
            time_point: Central time point
            window_sec: Time window in seconds (half-window before and after)
            
        Returns:
            Tuple of (time_start, time_end)
        """
        time_start = time_point - timedelta(seconds=window_sec)
        time_end = time_point + timedelta(seconds=window_sec)
        return time_start, time_end

    def _get_messages_in_window(
        self,
        session,
        chat_id: str,
        time_start: datetime,
        time_end: datetime
    ) -> List[Message]:
        """
        Get messages within a time window.
        
        Args:
            session: Database session
            chat_id: Chat ID to filter by
            time_start: Start of time window
            time_end: End of time window
            
        Returns:
            List of Message domain objects within the time window
        """
        query = session.query(MessageModel).filter(
            MessageModel.chat_id == chat_id,
            MessageModel.ts >= time_start,
            MessageModel.ts <= time_end
        ).order_by(MessageModel.ts)
        
        models = query.all()
        return [self._model_to_domain(model, session) for model in models]

    def get_context(self, chat_id: str, time_point: datetime, window_sec: int) -> List[Message]:
        """
        Get messages around a time point within a time window.

        Args:
            chat_id: Chat ID to filter by
            time_point: Central time point
            window_sec: Time window in seconds (half-window before and after)

        Returns:
            List of Message domain objects within the time window
        """
        session = self.db.get_session()
        try:
            time_start, time_end = self._calculate_time_window(time_point, window_sec)
            return self._get_messages_in_window(session, chat_id, time_start, time_end)
        finally:
            session.close()

    def count(self) -> int:
        """
        Get total count of messages.

        Returns:
            Total number of messages in the database
        """
        return self.db.count_messages()

    def _model_to_domain(self, model: MessageModel, session) -> Message:
        """
        Convert MessageModel to domain Message object.

        Args:
            model: MessageModel instance
            session: Database session for loading metadata

        Returns:
            Message domain object
        """
        # Load metadata if present
        meta = {}
        meta_model = session.query(MessageMetaModel).filter(
            MessageMetaModel.msg_id == model.msg_id
        ).first()
        
        if meta_model and meta_model.meta_json:
            try:
                meta = json.loads(meta_model.meta_json)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        return Message(
            id=model.msg_id,
            chat_id=model.chat_id,
            from_id=model.from_id or "",
            text=model.text,
            timestamp=model.ts,
            meta=meta
        )


class SqliteChunkStore:
    """SQLite adapter implementing ChunkStore protocol."""

    def __init__(self, database: Database):
        """
        Initialize the adapter with a Database instance.

        Args:
            database: Database instance from src.storage.db
        """
        self.db = database
        self._fts_index: Optional[SqliteFTSIndex] = None

    def _prepare_chunk_data(self, chunk: Chunk) -> Dict:
        """
        Prepare chunk data for database storage.
        
        Note: Embeddings are not stored in ChunkStore - they belong to VectorIndex.
        ChunkStore only stores text, metadata, and message relationships.
        
        Args:
            chunk: Chunk domain object
            
        Returns:
            Dictionary with prepared chunk data
        """
        # Prepare metadata JSON
        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None

        # Embeddings are not stored in ChunkStore - they belong to VectorIndex
        # embedding_json and embedding_dim are not part of ChunkStore responsibility

        # Extract message IDs and timestamps
        msg_id_start = None
        msg_id_end = None
        ts_from = None
        ts_to = None
        chat_id = None

        if chunk.msg_ids:
            msg_id_start = chunk.msg_ids[0]
            if len(chunk.msg_ids) > 1:
                msg_id_end = chunk.msg_ids[1]

        if chunk.valid_period:
            ts_from = chunk.valid_period[0]
            if len(chunk.valid_period) > 1:
                ts_to = chunk.valid_period[1]

        # Extract chat_id from metadata if available
        if chunk.metadata and "chat_id" in chunk.metadata:
            chat_id = str(chunk.metadata["chat_id"])

        return {
            "metadata_json": metadata_json,
            "msg_id_start": msg_id_start,
            "msg_id_end": msg_id_end,
            "ts_from": ts_from,
            "ts_to": ts_to,
            "chat_id": chat_id,
        }

    def _chunk_exists(self, session, chunk_id: str) -> Optional[ChunkModel]:
        """
        Check if chunk exists in database.
        
        Args:
            session: Database session
            chunk_id: Chunk ID
            
        Returns:
            ChunkModel instance if exists, None otherwise
        """
        return session.query(ChunkModel).filter(
            ChunkModel.id == chunk_id
        ).first()

    def _create_or_update_chunk(
        self,
        session,
        chunk: Chunk,
        chunk_data: Dict,
        existing: Optional[ChunkModel]
    ) -> bool:
        """
        Create or update chunk in database.
        
        Note: Embeddings are not stored in ChunkStore - they belong to VectorIndex.
        
        Args:
            session: Database session
            chunk: Chunk domain object
            chunk_data: Prepared chunk data dictionary
            existing: Existing ChunkModel instance or None
            
        Returns:
            True if new chunk was created, False if updated
        """
        if existing:
            # Update existing chunk
            existing.text = chunk.text
            existing.metadata_json = chunk_data["metadata_json"]
            # embedding_json and embedding_dim are not updated - they belong to VectorIndex
            existing.msg_id_start = chunk_data["msg_id_start"]
            existing.msg_id_end = chunk_data["msg_id_end"]
            existing.ts_from = chunk_data["ts_from"]
            existing.ts_to = chunk_data["ts_to"]
            if chunk_data["chat_id"]:
                existing.chat_id = chunk_data["chat_id"]
            return False
        else:
            # Create new chunk
            # Note: embedding_json and embedding_dim are not set here - they belong to VectorIndex
            # and should be managed directly through Database/ChunkModel, not through ChunkStore
            chunk_model = ChunkModel(
                id=chunk.id,
                text=chunk.text,
                metadata_json=chunk_data["metadata_json"],
                msg_id_start=chunk_data["msg_id_start"],
                msg_id_end=chunk_data["msg_id_end"],
                ts_from=chunk_data["ts_from"],
                ts_to=chunk_data["ts_to"],
                chat_id=chunk_data["chat_id"],
            )
            session.add(chunk_model)
            return True

    def save_batch(self, chunks: List[Chunk]) -> int:
        """
        Save a batch of chunks to the database.

        Args:
            chunks: List of Chunk domain objects

        Returns:
            Number of chunks saved
        """
        if not chunks:
            return 0

        saved_count = 0
        session = self.db.get_session()
        try:
            for chunk in chunks:
                # Prepare chunk data
                chunk_data = self._prepare_chunk_data(chunk)
                
                # Check if chunk exists
                existing = self._chunk_exists(session, chunk.id)
                
                # Create or update chunk
                if self._create_or_update_chunk(session, chunk, chunk_data, existing):
                    saved_count += 1

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

        # FTS5 indexes are updated automatically via triggers
        # But we ensure FTS tables exist
        if self._fts_index is None:
            self._fts_index = SqliteFTSIndex(self.db)

        return saved_count

    def get_fts_index(self) -> FTSIndex:
        """
        Get FTS5 index for chunks.

        Returns:
            FTSIndex instance for searching chunks
        """
        if self._fts_index is None:
            self._fts_index = SqliteFTSIndex(self.db)
        return self._fts_index

    def get_by_ids(self, ids: List[str]) -> List[Chunk]:
        """
        Get chunks by their IDs.

        Args:
            ids: List of chunk IDs

        Returns:
            List of Chunk domain objects
        """
        if not ids:
            return []

        # Fetch chunks from database
        models = self._fetch_chunks_from_db(ids)
        
        # Convert to domain objects
        return self._convert_to_domain_chunks(models)
    
    def _fetch_chunks_from_db(self, ids: List[str]) -> List[ChunkModel]:
        """
        Fetch chunk models from database by IDs.
        
        Args:
            ids: List of chunk IDs
            
        Returns:
            List of ChunkModel instances
        """
        session = self.db.get_session()
        try:
            # Query chunks by IDs
            models = session.query(ChunkModel).filter(
                ChunkModel.id.in_(ids)
            ).all()
            return models
        finally:
            session.close()
    
    def _convert_to_domain_chunks(self, models: List[ChunkModel]) -> List[Chunk]:
        """
        Convert ChunkModel instances to domain Chunk objects.
        
        Args:
            models: List of ChunkModel instances
            
        Returns:
            List of Chunk domain objects
        """
        return [self._model_to_domain(model) for model in models]

    def _update_chunk_topics(self, session, chunk: ChunkModel, topic_update: TopicUpdate) -> None:
        """
        Update topic assignments for a single chunk.
        
        Args:
            session: Database session
            chunk: ChunkModel instance to update
            topic_update: TopicUpdate object with topic information
        """
        # Extract topic IDs from TopicUpdate
        # TopicUpdate has topic_ids list - we need to map to topic_l1_id and topic_l2_id
        # For now, we'll store topic IDs in metadata and try to extract L1/L2 IDs
        topic_l1_id = None
        topic_l2_id = None

        # Parse topic_ids - assume format like "l1:123" or "l2:456" or just integer IDs
        for topic_id_str in topic_update.topic_ids:
            if topic_id_str.startswith("l1:"):
                try:
                    topic_l1_id = int(topic_id_str.split(":", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif topic_id_str.startswith("l2:"):
                try:
                    topic_l2_id = int(topic_id_str.split(":", 1)[1])
                except (ValueError, IndexError):
                    pass
            else:
                # Try to parse as integer - assume L1 if not specified
                try:
                    topic_l1_id = int(topic_id_str)
                except ValueError:
                    pass

        # Update topic assignments
        if topic_l1_id is not None:
            chunk.topic_l1_id = topic_l1_id
        if topic_l2_id is not None:
            chunk.topic_l2_id = topic_l2_id

        # Update metadata if provided
        if topic_update.metadata:
            existing_meta = {}
            if chunk.metadata_json:
                try:
                    existing_meta = json.loads(chunk.metadata_json)
                except (json.JSONDecodeError, TypeError):
                    existing_meta = {}
            
            existing_meta.update(topic_update.metadata)
            chunk.metadata_json = json.dumps(existing_meta)

    def _batch_update_topics(self, session, updates: Dict[str, TopicUpdate]) -> None:
        """
        Perform batch update of topic assignments.
        
        Args:
            session: Database session
            updates: Dictionary mapping chunk_id to TopicUpdate
        """
        for chunk_id, topic_update in updates.items():
            chunk = session.query(ChunkModel).filter(
                ChunkModel.id == chunk_id
            ).first()

            if not chunk:
                continue

            self._update_chunk_topics(session, chunk, topic_update)

    def update_topics(self, updates: Dict[str, TopicUpdate]) -> None:
        """
        Update topic assignments for chunks.

        Args:
            updates: Dictionary mapping chunk_id to TopicUpdate
        """
        if not updates:
            return

        session = self.db.get_session()
        try:
            self._batch_update_topics(session, updates)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear(self) -> None:
        """
        Clear all chunks from the database.
        """
        self.db.clear()

    def _model_to_domain(self, model: ChunkModel) -> Chunk:
        """
        Convert ChunkModel to domain Chunk object.

        Note: Embeddings are not loaded from ChunkStore - they belong to VectorIndex.
        The embedding field in Chunk domain object will be None when loaded from ChunkStore.
        To get embeddings, use VectorIndex.get_embeddings_by_ids().

        Args:
            model: ChunkModel instance

        Returns:
            Chunk domain object (with embedding=None, as embeddings are in VectorIndex)
        """
        # Parse metadata
        metadata = {}
        if model.metadata_json:
            try:
                metadata = json.loads(model.metadata_json)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # Embeddings are not loaded from ChunkStore - they belong to VectorIndex
        # If you need embeddings, use VectorIndex.get_embeddings_by_ids()
        embedding = None

        # Build msg_ids tuple (must be 2-tuple: (start_id, end_id))
        msg_ids = None
        if model.msg_id_start:
            end_id = model.msg_id_end if model.msg_id_end else model.msg_id_start
            msg_ids = (model.msg_id_start, end_id)

        # Build valid_period tuple (must be 2-tuple: (from, to))
        valid_period = None
        if model.ts_from:
            to_ts = model.ts_to if model.ts_to else model.ts_from
            valid_period = (model.ts_from, to_ts)

        return Chunk(
            id=model.id,
            text=model.text,
            msg_ids=msg_ids,
            valid_period=valid_period,
            embedding=embedding,  # Always None - embeddings are in VectorIndex
            metadata=metadata
        )
