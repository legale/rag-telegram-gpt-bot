from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from typing import List, Optional, Tuple, Any
import json
from src.lib.syslog2 import *

Base = declarative_base()

# ============================================================================
# Models
# ============================================================================

class MessageModel(Base):
    """Stores raw messages from chat."""
    __tablename__ = 'messages'
    
    msg_id = Column(String, primary_key=True)
    chat_id = Column(String, nullable=False, index=True)
    ts = Column(DateTime, nullable=False, index=True)
    from_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)


class MessageMetaModel(Base):
    """Stores additional metadata for messages."""
    __tablename__ = 'message_meta'
    
    msg_id = Column(String, ForeignKey('messages.msg_id', ondelete='CASCADE'), primary_key=True)
    meta_json = Column(Text, nullable=True)  # JSON with additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class ChunkModel(Base):
    """Stores text chunks with message reference and topic assignments."""
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, nullable=True)
    
    # Message-based chunking fields
    chat_id = Column(String, nullable=True, index=True)
    msg_id_start = Column(String, ForeignKey('messages.msg_id', ondelete='SET NULL'), nullable=True)
    msg_id_end = Column(String, ForeignKey('messages.msg_id', ondelete='SET NULL'), nullable=True)
    # Raw msg_id fields (without chat_id prefix) for easier access
    msg_id_start_raw = Column(String, nullable=True)
    msg_id_end_raw = Column(String, nullable=True)
    ts_from = Column(DateTime, nullable=True, index=True)
    ts_to = Column(DateTime, nullable=True)
    
    # Topic assignments removed - clustering is deprecated
    
    # Embedding indicator
    embedding_dim = Column(Integer, nullable=True, index=True)
    # Embedding storage (JSON array of floats)
    embedding_json = Column(Text, nullable=True)


class UserModel(Base):
    """Stores user information and aliases."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False, unique=True, index=True)
    aliases = Column(Text, nullable=True)  # JSON list of aliases

# ============================================================================
# Database Class
# ============================================================================

class Database:
    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("db_url must be provided")
        self.db_url = db_url
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Simple auto-migration for dev environment (adding new columns if missing)
        self._ensure_schema()
        
    def _create_fts5_tables(self, conn) -> None:
        """
        Create FTS5 tables for messages and chunks.
        
        Args:
            conn: Database connection
        """
        from sqlalchemy import text
        # Create messages_fts table if it doesn't exist
        conn.execute(text("""
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
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                chat_id UNINDEXED,
                text,
                content='chunks',
                content_rowid='rowid'
            )
        """))

    def _create_fts5_triggers(self, conn) -> None:
        """
        Create triggers to keep FTS5 tables in sync with main tables.
        
        Args:
            conn: Database connection
        """
        from sqlalchemy import text
        # Messages triggers
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
            END
        """))
        
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE msg_id = old.msg_id;
            END
        """))
        
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                DELETE FROM messages_fts WHERE msg_id = old.msg_id;
                INSERT INTO messages_fts(msg_id, chat_id, from_id, ts, text)
                VALUES (new.msg_id, new.chat_id, new.from_id, new.ts, new.text);
            END
        """))
        
        # Chunks triggers
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(id, chat_id, text)
                VALUES (new.id, new.chat_id, new.text);
            END
        """))
        
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE id = old.id;
            END
        """))
        
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE id = old.id;
                INSERT INTO chunks_fts(id, chat_id, text)
                VALUES (new.id, new.chat_id, new.text);
            END
        """))

    def _check_and_add_columns(self, conn) -> None:
        """
        Check and add missing columns to chunks table.
        
        Args:
            conn: Database connection
        """
        from sqlalchemy import text
        # Check chunks table columns
        try:
            # We can't easily check all at once, so we try accessing one.
            # If chat_id is missing, we assume Phase 14.2 columns are missing.
            conn.execute(text("SELECT chat_id FROM chunks LIMIT 1"))
        except Exception:
            # Phase 14.2 columns missing
            try:
                conn.execute(text("ALTER TABLE chunks ADD COLUMN chat_id VARCHAR"))
                conn.execute(text("ALTER TABLE chunks ADD COLUMN msg_id_start VARCHAR"))
                conn.execute(text("ALTER TABLE chunks ADD COLUMN msg_id_end VARCHAR"))
                conn.execute(text("ALTER TABLE chunks ADD COLUMN ts_from DATETIME"))
                conn.execute(text("ALTER TABLE chunks ADD COLUMN ts_to DATETIME"))
                conn.commit()
            except Exception as e:
                syslog2(LOG_WARNING, "schema update warning (chunks 14.2)", error=str(e))

        # topic_l1_id and topic_l2_id removed - clustering is deprecated
        
        # Check chunks table for msg_id_start_raw/msg_id_end_raw (refactoring)
        try:
            conn.execute(text("SELECT msg_id_start_raw FROM chunks LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE chunks ADD COLUMN msg_id_start_raw VARCHAR"))
                conn.execute(text("ALTER TABLE chunks ADD COLUMN msg_id_end_raw VARCHAR"))
                conn.commit()
            except Exception as e:
                syslog2(LOG_WARNING, "schema update warning (chunks raw msg_id)", error=str(e))
        
        # Check chunks table for embedding_dim (refactoring)
        try:
            conn.execute(text("SELECT embedding_dim FROM chunks LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE chunks ADD COLUMN embedding_dim INTEGER"))
                conn.commit()
            except Exception as e:
                syslog2(LOG_WARNING, "schema update warning (chunks embedding_dim)", error=str(e))

    def _ensure_schema(self):
        """Checks for new columns and adds them if missing (SQLite specific)."""
        from sqlalchemy import text
        with self.engine.connect() as conn:
            # Ensure FTS5 tables exist (for hybrid retrieval)
            try:
                self._create_fts5_tables(conn)
                self._create_fts5_triggers(conn)
                conn.commit()
            except Exception as e:
                # FTS5 might not be available, log warning but continue
                syslog2(LOG_DEBUG, "fts5 tables creation skipped", error=str(e))
            
            # Check and add missing columns
            self._check_and_add_columns(conn)
            
            # Check chunks table for embedding_json (refactoring - stage2)
            try:
                conn.execute(text("SELECT embedding_json FROM chunks LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN embedding_json TEXT"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (chunks embedding_json)", error=str(e))
            
            # topics_l1 and topics_l2 tables removed - clustering is deprecated
            
            # Create message_meta table if it doesn't exist (refactoring - stage0)
            try:
                conn.execute(text("SELECT msg_id FROM message_meta LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS message_meta (
                            msg_id VARCHAR PRIMARY KEY,
                            meta_json TEXT,
                            created_at DATETIME,
                            FOREIGN KEY(msg_id) REFERENCES messages(msg_id) ON DELETE CASCADE
                        )
                    """))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (message_meta)", error=str(e))
            
            # Create users table if it doesn't exist
            try:
                conn.execute(text("SELECT username FROM users LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username VARCHAR NOT NULL UNIQUE,
                            aliases TEXT
                        )
                    """))
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (users)", error=str(e))
        
    def get_session(self):
        return self.Session()

    # ========================================================================
    # Chunk Methods
    # ========================================================================

    def count_chunks(self) -> int:
        """Returns number of stored chunks."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).count()
        finally:
            session.close()

    def clear(self) -> int:
        """Deletes all records from the chunks table and returns removed count."""
        session = self.get_session()
        try:
            deleted = session.query(ChunkModel).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_messages(self) -> int:
        """Delete all messages (stage0)."""
        session = self.get_session()
        try:
            deleted = session.query(MessageModel).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_chunk_text(self, chunk_id: str) -> str:
        """Helper to get text for a chunk."""
        session = self.get_session()
        try:
            chunk = session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
            return chunk.text if chunk else ""
        finally:
            session.close()

    def add_chunk_with_messages(
        self,
        chunk_id: str,
        text: str,
        chat_id: str,
        msg_id_start: str,
        msg_id_end: str,
        ts_from: datetime,
        ts_to: datetime,
        metadata_json: Optional[str] = None
    ) -> None:
        """Add a chunk with message references."""
        session = self.get_session()
        try:
            chunk = ChunkModel(
                id=chunk_id,
                text=text,
                chat_id=chat_id,
                msg_id_start=msg_id_start,
                msg_id_end=msg_id_end,
                ts_from=ts_from,
                ts_to=ts_to,
                metadata_json=metadata_json
            )
            session.add(chunk)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_chunk(self, chunk_id: str) -> Optional[ChunkModel]:
        """Get a single chunk by ID."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
        finally:
            session.close()

    def _extract_chat_id(self, chunk: ChunkModel) -> Optional[int]:
        """
        Extract chat_id from chunk.
        
        Args:
            chunk: ChunkModel instance
            
        Returns:
            chat_id as int, or None if not available or invalid
        """
        if not chunk.chat_id:
            return None
        
        try:
            # chat_id is stored as string, convert to int
            return int(chunk.chat_id)
        except (ValueError, TypeError):
            return None
    
    def _extract_msg_id(self, chunk: ChunkModel) -> Optional[int]:
        """
        Extract msg_id from chunk.msg_id_start.
        msg_id_start format: "{chat_id}_{msg_id}" or just "{msg_id}"
        
        Args:
            chunk: ChunkModel instance
            
        Returns:
            msg_id as int, or None if not available or invalid
        """
        if not chunk.msg_id_start:
            return None
        
        try:
            # Extract the numeric part after the underscore
            parts = chunk.msg_id_start.split('_', 1)
            if len(parts) > 1:
                return int(parts[1])
            else:
                # If no underscore, try to parse the whole string
                return int(chunk.msg_id_start)
        except (ValueError, TypeError, IndexError):
            return None
    
    def _extract_chat_username(self, chunk: ChunkModel) -> Optional[str]:
        """
        Extract chat_username from chunk.metadata_json.
        
        Args:
            chunk: ChunkModel instance
            
        Returns:
            chat_username as string, or None if not available or invalid
        """
        if not chunk.metadata_json:
            return None
        
        try:
            meta = json.loads(chunk.metadata_json)
            if isinstance(meta, dict):
                return meta.get("chat_username")
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def get_chunk_link_info(self, chunk_id: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Get link information for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Tuple of (chat_id, msg_id, chat_username)
            Returns (None, None, None) if chunk not found or missing required fields
        """
        session = self.get_session()
        try:
            chunk = session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
            if chunk is None:
                return (None, None, None)
            
            chat_id = self._extract_chat_id(chunk)
            msg_id = self._extract_msg_id(chunk)
            chat_username = self._extract_chat_username(chunk)
            
            return (chat_id, msg_id, chat_username)
        finally:
            session.close()

    def get_message_by_id(self, msg_id: str) -> Optional[MessageModel]:
        """
        Get a message by its ID.
        
        Args:
            msg_id: Message ID (can be composite format "{chat_id}_{msg_id}")
            
        Returns:
            MessageModel instance or None if not found
        """
        session = self.get_session()
        try:
            return session.query(MessageModel).filter(MessageModel.msg_id == msg_id).first()
        finally:
            session.close()

    def _get_start_message(self, session, msg_id_start: str) -> Optional[MessageModel]:
        """
        Get start message by msg_id.
        
        Args:
            session: Database session
            msg_id_start: Start message ID
            
        Returns:
            MessageModel instance or None if not found
        """
        return session.query(MessageModel).filter(
            MessageModel.msg_id == msg_id_start
        ).first()
    
    def _get_end_message(self, session, msg_id_end: str) -> Optional[MessageModel]:
        """
        Get end message by msg_id.
        
        Args:
            session: Database session
            msg_id_end: End message ID
            
        Returns:
            MessageModel instance or None if not found
        """
        return session.query(MessageModel).filter(
            MessageModel.msg_id == msg_id_end
        ).first()
    
    def _get_messages_in_range(self, session, chat_id: int, start_msg: MessageModel, end_msg: MessageModel) -> List[MessageModel]:
        """
        Get all messages between start and end (inclusive) by timestamp.
        
        Args:
            session: Database session
            chat_id: Chat ID
            start_msg: Start message
            end_msg: End message
            
        Returns:
            List of MessageModel instances, ordered by timestamp
        """
        return session.query(MessageModel).filter(
            MessageModel.chat_id == chat_id,
            MessageModel.ts >= start_msg.ts,
            MessageModel.ts <= end_msg.ts
        ).order_by(MessageModel.ts).all()
    
    def get_messages_by_chunk(self, chunk_id: str) -> List[MessageModel]:
        """
        Get all messages included in a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            List of MessageModel instances, ordered by timestamp
        """
        session = self.get_session()
        try:
            chunk = session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
            if chunk is None or not chunk.msg_id_start:
                return []
            
            # Get start message
            start_msg = self._get_start_message(session, chunk.msg_id_start)
            if not start_msg:
                return []
            
            # If msg_id_end is specified, get messages in range
            if chunk.msg_id_end:
                end_msg = self._get_end_message(session, chunk.msg_id_end)
                if end_msg:
                    # Get all messages between start and end (inclusive) by timestamp
                    return self._get_messages_in_range(session, chunk.chat_id, start_msg, end_msg)
                else:
                    # End message not found, return just start message
                    return [start_msg]
            else:
                # Only start message specified
                return [start_msg]
        finally:
            session.close()

    # ========================================================================
    # Message Methods
    # ========================================================================

    def add_message(self, msg_id: str, chat_id: str, ts: datetime, from_id: str, text: str) -> None:
        """Add a message to the database."""
        session = self.get_session()
        try:
            message = MessageModel(
                msg_id=msg_id,
                chat_id=chat_id,
                ts=ts,
                from_id=from_id,
                text=text
            )
            session.add(message)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def _get_existing_msg_ids(self, session, msg_ids: List[str]) -> set:
        """
        Get set of existing message IDs from database.
        
        Args:
            session: Database session
            msg_ids: List of message IDs to check
            
        Returns:
            Set of existing message IDs
        """
        existing_msg_ids = set()
        batch_size = 900  # Safe limit below SQLite's 999 parameter limit
        
        for i in range(0, len(msg_ids), batch_size):
            batch = msg_ids[i:i + batch_size]
            batch_existing = session.query(MessageModel.msg_id).filter(
                MessageModel.msg_id.in_(batch)
            ).all()
            existing_msg_ids.update(row[0] for row in batch_existing)
        
        return existing_msg_ids

    def _filter_new_messages(self, messages: List[dict], existing_msg_ids: set) -> List[dict]:
        """
        Filter out messages that already exist in database.
        
        Args:
            messages: List of message dictionaries
            existing_msg_ids: Set of existing message IDs
            
        Returns:
            List of new messages (not in database)
        """
        return [msg for msg in messages if msg["msg_id"] not in existing_msg_ids]

    def add_messages_batch(self, messages: List[dict]) -> int:
        """
        Add multiple messages to the database in one transaction.
        Skips messages that already exist (duplicate msg_id).
        Args:
            messages: List of dictionaries matching MessageModel fields.
        Returns:
            Number of messages actually inserted (excluding duplicates).
        """
        if not messages:
            return 0
        
        session = self.get_session()
        try:
            # Get existing msg_ids to avoid duplicates
            msg_ids_to_insert = [msg["msg_id"] for msg in messages]
            existing_msg_ids = self._get_existing_msg_ids(session, msg_ids_to_insert)
            
            # Filter out messages that already exist
            new_messages = self._filter_new_messages(messages, existing_msg_ids)
            
            if not new_messages:
                # All messages already exist
                return 0
            
            # Insert only new messages
            models = [MessageModel(**msg) for msg in new_messages]
            session.add_all(models)
            session.commit()
            return len(new_messages)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_messages(self, limit: Optional[int] = None) -> List[MessageModel]:
        """Get all messages, optionally limited."""
        session = self.get_session()
        try:
            query = session.query(MessageModel).order_by(MessageModel.ts)
            if limit:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()

    def get_messages_by_chat(self, chat_id: str, limit: Optional[int] = None) -> List[MessageModel]:
        """Get messages for a specific chat, sorted by timestamp."""
        session = self.get_session()
        try:
            query = session.query(MessageModel).filter(
                MessageModel.chat_id == chat_id
            ).order_by(MessageModel.ts)
            if limit:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()

    def count_messages(self, chat_id: Optional[str] = None) -> int:
        """Count messages, optionally filtered by chat_id."""
        session = self.get_session()
        try:
            query = session.query(MessageModel)
            if chat_id:
                query = query.filter(MessageModel.chat_id == chat_id)
            return query.count()
        finally:
            session.close()

    def _build_fts_where_clause(
        self,
        fts_query: str,
        chat_id: Optional[str] = None,
        from_id: Optional[str] = None,
        ts_min: Optional[datetime] = None,
        ts_max: Optional[datetime] = None
    ) -> Tuple[str, dict]:
        """
        Build WHERE clause for FTS5 search.
        
        Args:
            fts_query: FTS5 MATCH query string
            chat_id: Optional chat_id filter
            from_id: Optional from_id filter
            ts_min: Optional minimum timestamp filter
            ts_max: Optional maximum timestamp filter
            
        Returns:
            Tuple of (where_sql, params_dict)
        """
        where_parts = ["messages_fts MATCH :query"]
        params = {"query": fts_query}
        
        if chat_id:
            where_parts.append("messages_fts.chat_id = :chat_id")
            params["chat_id"] = chat_id
        
        if from_id:
            where_parts.append("messages_fts.from_id = :from_id")
            params["from_id"] = from_id
        
        if ts_min:
            where_parts.append("messages_fts.ts >= :ts_min")
            params["ts_min"] = ts_min
        
        if ts_max:
            where_parts.append("messages_fts.ts <= :ts_max")
            params["ts_max"] = ts_max
        
        where_sql = "WHERE " + " AND ".join(where_parts)
        return where_sql, params
    
    def _execute_fts_query(
        self,
        where_sql: str,
        params: dict,
        top_k: int
    ) -> List[Tuple]:
        """
        Execute FTS5 search query.
        
        Args:
            where_sql: WHERE clause SQL
            params: Query parameters dictionary
            top_k: Number of results to return
            
        Returns:
            List of raw result rows (msg_id, score)
        """
        from sqlalchemy import text
        session = self.get_session()
        try:
            params["top_k"] = top_k
            sql = f"""
                SELECT 
                    messages_fts.msg_id,
                    bm25(messages_fts) as score
                FROM messages_fts
                {where_sql}
                ORDER BY bm25(messages_fts) ASC
                LIMIT :top_k
            """
            
            result = session.execute(text(sql), params)
            return result.fetchall()
        finally:
            session.close()
    
    def _convert_fts_results(self, rows: List[Tuple]) -> List[Tuple[str, float]]:
        """
        Convert raw FTS query results to (msg_id, score) tuples.
        
        Args:
            rows: Raw result rows from database
            
        Returns:
            List of (msg_id, score) tuples
        """
        return [(str(row[0]), float(row[1])) for row in rows]
    
    def fts_search(
        self,
        fts_query: str,
        top_k: int,
        chat_id: Optional[str] = None,
        from_id: Optional[str] = None,
        ts_min: Optional[datetime] = None,
        ts_max: Optional[datetime] = None
    ) -> List[Tuple[str, float]]:
        """
        FTS5 search in messages_fts table using bm25 scoring.
        
        Args:
            fts_query: FTS5 MATCH query string
            top_k: Number of results to return
            chat_id: Optional chat_id filter
            from_id: Optional from_id filter
            ts_min: Optional minimum timestamp filter
            ts_max: Optional maximum timestamp filter
            
        Returns:
            List of (msg_id, score) tuples, sorted by score ASC (lower is better in bm25)
        """
        if not fts_query or not fts_query.strip():
            return []
        
        try:
            where_sql, params = self._build_fts_where_clause(
                fts_query, chat_id, from_id, ts_min, ts_max
            )
            rows = self._execute_fts_query(where_sql, params, top_k)
            return self._convert_fts_results(rows)
            
        except Exception as e:
            syslog2(LOG_ERR, "fts_search failed", fts_query=fts_query, error=str(e))
            return []

    # ========================================================================
    # Topic L1 Methods - REMOVED (clustering is deprecated)
    # ========================================================================

    # ========================================================================
    # User Methods
    # ========================================================================

    def add_user(self, username: str, aliases: List[str] = None) -> None:
        """
        Add a new user or update existing one.
        
        Args:
            username: Username (primary key)
            aliases: Optional list of aliases
        """
        session = self.get_session()
        try:
            user = session.query(UserModel).filter(UserModel.username == username).first()
            if not user:
                user = UserModel(username=username)
                session.add(user)
            
            if aliases is not None:
                user.aliases = json.dumps(aliases)
                
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_user(self, username: str) -> Optional[UserModel]:
        """
        Get user by username.
        
        Args:
            username: Username to find
            
        Returns:
            UserModel or None
        """
        session = self.get_session()
        try:
            return session.query(UserModel).filter(UserModel.username == username).first()
        finally:
            session.close()

    def update_user_aliases(self, username: str, aliases: List[str]) -> None:
        """
        Update aliases for a user.
        
        Args:
            username: Username to update
            aliases: List of new aliases
        """
        self.add_user(username, aliases)

    def get_all_users(self) -> List[UserModel]:
        """Get all users."""
        session = self.get_session()
        try:
            return session.query(UserModel).all()
        finally:
            session.close()

    def get_messages_by_user(self, username: str, limit: Optional[int] = None, date_filter: Optional[datetime] = None) -> List[MessageModel]:
        """
        Get messages for a specific user, including aliases.
        
        Args:
            username: Username of the user
            limit: Optional limit on number of messages
            date_filter: Optional datetime to filter messages after
            
        Returns:
            List of MessageModel objects
        """
        user = self.get_user(username)
        aliases = []
        if user and user.aliases:
            try:
                aliases = json.loads(user.aliases)
            except Exception:
                pass
        
        # Search by username and all aliases
        search_names = [username] + aliases
        
        session = self.get_session()
        try:
            query = session.query(MessageModel).filter(
                MessageModel.from_id.in_(search_names)
            )
            
            if date_filter:
                query = query.filter(MessageModel.ts >= date_filter)
                
            query = query.order_by(MessageModel.ts.desc())
            
            if limit:
                query = query.limit(limit)
                
            results = query.all()
            # Sort back to chronological order
            return sorted(results, key=lambda x: x.ts)
        finally:
            session.close()

    def get_neighbor_messages(self, target_msg: MessageModel, window_count: int = 20, max_tokens: int = 1000) -> List[MessageModel]:
        """
        Get neighbor messages around a target message within the same day.
        
        Args:
            target_msg: Target message object
            window_count: Max messages to fetch (before + after)
            max_tokens: Max total tokens allowed (approximate)
            
        Returns:
            List of MessageModel objects including target, sorted by time
        """
        session = self.get_session()
        try:
            # Defines start and end of the day for the target message
            day_start = target_msg.ts.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = target_msg.ts.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Fetch surrounding messages from the same chat on the same day
            # Context before
            before = session.query(MessageModel).filter(
                MessageModel.chat_id == target_msg.chat_id,
                MessageModel.ts >= day_start,
                MessageModel.ts < target_msg.ts
            ).order_by(MessageModel.ts.desc()).limit(window_count).all()
            
            # Context after
            after = session.query(MessageModel).filter(
                MessageModel.chat_id == target_msg.chat_id,
                MessageModel.ts > target_msg.ts,
                MessageModel.ts <= day_end
            ).order_by(MessageModel.ts.asc()).limit(window_count).all()
            
            # Combine and sort (before is desc, so reverse it)
            context = sorted(before, key=lambda x: x.ts) + [target_msg] + sorted(after, key=lambda x: x.ts)
            
            # Simple token limit enforcement (approx 4 chars per token)
            current_tokens = 0
            final_context = []
            
            # Find index of target in full list
            target_idx = -1
            for i, m in enumerate(context):
                if m.msg_id == target_msg.msg_id:
                    target_idx = i
                    break
            
            if target_idx == -1:
                return [target_msg]

            # Expand from center
            final_context.append(context[target_idx])
            current_tokens += len(context[target_idx].text) // 4
            
            left = target_idx - 1
            right = target_idx + 1
            
            expand_left = True
            expand_right = True
            
            while (expand_left or expand_right) and current_tokens < max_tokens:
                # Try adding left neighbor
                if expand_left:
                    if left >= 0:
                        msg = context[left]
                        tokens = len(msg.text) // 4
                        if current_tokens + tokens <= max_tokens:
                            final_context.insert(0, msg)
                            current_tokens += tokens
                            left -= 1
                        else:
                            expand_left = False
                    else:
                        expand_left = False
                
                # Try adding right neighbor
                if expand_right:
                    if right < len(context):
                        msg = context[right]
                        tokens = len(msg.text) // 4
                        if current_tokens + tokens <= max_tokens:
                            final_context.append(msg)
                            current_tokens += tokens
                            right += 1
                        else:
                            expand_right = False
                    else:
                        expand_right = False
                    
            return final_context
        finally:
            session.close()

    def get_database_info(self) -> dict:
        """
        Get statistics for all tables in the database.
        Returns a dictionary mapping table names to record counts.
        """
        session = self.get_session()
        try:
            info = {}
            
            # Main tables (may not exist if database is new)
            try:
                info['messages'] = session.query(MessageModel).count()
            except Exception:
                info['messages'] = 0
            
            try:
                info['chunks'] = session.query(ChunkModel).count()
            except Exception:
                info['chunks'] = 0
            
            # topics_l1 and topics_l2 removed - clustering is deprecated
            
            return info
        finally:
            session.close()
