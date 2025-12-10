from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from typing import List, Optional, Tuple, Any
import json
from src.core.syslog2 import *

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
    
    # Topic assignments
    topic_l1_id = Column(Integer, ForeignKey('topics_l1.id', ondelete='SET NULL'), nullable=True, index=True)
    topic_l2_id = Column(Integer, ForeignKey('topics_l2.id', ondelete='SET NULL'), nullable=True, index=True)
    
    # Embedding indicator
    embedding_dim = Column(Integer, nullable=True, index=True)
    # Embedding storage (JSON array of floats)
    embedding_json = Column(Text, nullable=True)

    # Relationships
    topic_l1 = relationship("TopicL1Model", back_populates="chunks")
    topic_l2 = relationship("TopicL2Model", back_populates="chunks")


class TopicL1Model(Base):
    """L1 topics (fine-grained topics from chunk clustering)."""
    __tablename__ = 'topics_l1'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    descr = Column(Text, nullable=False)
    parent_l2_id = Column(Integer, ForeignKey('topics_l2.id', ondelete='SET NULL'), nullable=True, index=True)
    chunk_count = Column(Integer, nullable=False, default=0)
    msg_count = Column(Integer, nullable=False, default=0)
    ts_from = Column(DateTime, nullable=True)
    ts_to = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Center vector storage (JSON array of floats)
    center_vec_json = Column(Text, nullable=True)

    # Relationships
    chunks = relationship("ChunkModel", back_populates="topic_l1")
    parent_l2 = relationship("TopicL2Model", back_populates="topics_l1")


class TopicL2Model(Base):
    """L2 topics (super-topics from L1 clustering)."""
    __tablename__ = 'topics_l2'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    descr = Column(Text, nullable=False)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Center vector storage (JSON array of floats)
    center_vec_json = Column(Text, nullable=True)

    # Relationships
    chunks = relationship("ChunkModel", back_populates="topic_l2")
    topics_l1 = relationship("TopicL1Model", back_populates="parent_l2")


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
        
    def _ensure_schema(self):
        """Checks for new columns and adds them if missing (SQLite specific)."""
        from sqlalchemy import text
        with self.engine.connect() as conn:
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

            # Check chunks table for topic_l1_id (Phase 14.3)
            try:
                conn.execute(text("SELECT topic_l1_id FROM chunks LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN topic_l1_id INTEGER"))
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN topic_l2_id INTEGER"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (chunks 14.3)", error=str(e))

            # Check topics_l1 table for parent_l2_id
            try:
                conn.execute(text("SELECT parent_l2_id FROM topics_l1 LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE topics_l1 ADD COLUMN parent_l2_id INTEGER"))
                    conn.commit()
                except Exception as e:
                     syslog2(LOG_WARNING, "schema update warning (topics_l1)", error=str(e))
            
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
            
            # Check chunks table for embedding_json (refactoring - stage2)
            try:
                conn.execute(text("SELECT embedding_json FROM chunks LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN embedding_json TEXT"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (chunks embedding_json)", error=str(e))
            
            # Check topics_l1 table for center_vec_json (refactoring - stage4)
            try:
                conn.execute(text("SELECT center_vec_json FROM topics_l1 LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE topics_l1 ADD COLUMN center_vec_json TEXT"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (topics_l1 center_vec_json)", error=str(e))
            
            # Check topics_l2 table for center_vec_json (refactoring - stage6)
            try:
                conn.execute(text("SELECT center_vec_json FROM topics_l2 LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE topics_l2 ADD COLUMN center_vec_json TEXT"))
                    conn.commit()
                except Exception as e:
                    syslog2(LOG_WARNING, "schema update warning (topics_l2 center_vec_json)", error=str(e))
            
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

    def clear_chunk_topic_l1_assignments(self) -> int:
        """Clear topic_l1_id assignments in chunks (stage3)."""
        session = self.get_session()
        try:
            updated = session.query(ChunkModel).update({ChunkModel.topic_l1_id: None}, synchronize_session=False)
            session.commit()
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_chunk_topic_l2_assignments(self) -> int:
        """Clear topic_l2_id assignments in chunks (stage4)."""
        session = self.get_session()
        try:
            updated = session.query(ChunkModel).update({ChunkModel.topic_l2_id: None}, synchronize_session=False)
            session.commit()
            return updated
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

    def update_chunk_topics(self, chunk_id: str, topic_l1_id: Optional[int], topic_l2_id: Optional[int]) -> None:
        """Update topic assignments for a chunk."""
        session = self.get_session()
        try:
            chunk = session.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
            if chunk:
                chunk.topic_l1_id = topic_l1_id
                chunk.topic_l2_id = topic_l2_id
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
            
            # Extract chat_id
            chat_id = None
            if chunk.chat_id:
                try:
                    # chat_id is stored as string, convert to int
                    chat_id = int(chunk.chat_id)
                except (ValueError, TypeError):
                    pass
            
            # Extract msg_id from msg_id_start
            # msg_id_start format: "{chat_id}_{msg_id}"
            msg_id = None
            if chunk.msg_id_start:
                try:
                    # Extract the numeric part after the underscore
                    parts = chunk.msg_id_start.split('_', 1)
                    if len(parts) > 1:
                        msg_id = int(parts[1])
                    else:
                        # If no underscore, try to parse the whole string
                        msg_id = int(chunk.msg_id_start)
                except (ValueError, TypeError, IndexError):
                    pass
            
            # Extract chat_username from metadata_json (optional)
            chat_username = None
            if chunk.metadata_json:
                try:
                    meta = json.loads(chunk.metadata_json)
                    if isinstance(meta, dict):
                        chat_username = meta.get("chat_username")
                except (json.JSONDecodeError, TypeError):
                    pass
            
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
            start_msg = session.query(MessageModel).filter(
                MessageModel.msg_id == chunk.msg_id_start
            ).first()
            
            if not start_msg:
                return []
            
            # If msg_id_end is specified, get messages in range
            if chunk.msg_id_end:
                end_msg = session.query(MessageModel).filter(
                    MessageModel.msg_id == chunk.msg_id_end
                ).first()
                
                if end_msg:
                    # Get all messages between start and end (inclusive) by timestamp
                    messages = session.query(MessageModel).filter(
                        MessageModel.chat_id == chunk.chat_id,
                        MessageModel.ts >= start_msg.ts,
                        MessageModel.ts <= end_msg.ts
                    ).order_by(MessageModel.ts).all()
                    return messages
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
            # Batch the query to avoid SQLite parameter limits (999 max)
            msg_ids_to_insert = [msg["msg_id"] for msg in messages]
            existing_msg_ids = set()
            batch_size = 900  # Safe limit below SQLite's 999 parameter limit
            
            for i in range(0, len(msg_ids_to_insert), batch_size):
                batch = msg_ids_to_insert[i:i + batch_size]
                batch_existing = session.query(MessageModel.msg_id).filter(
                    MessageModel.msg_id.in_(batch)
                ).all()
                existing_msg_ids.update(row[0] for row in batch_existing)
            
            # Filter out messages that already exist
            new_messages = [msg for msg in messages if msg["msg_id"] not in existing_msg_ids]
            
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

    # ========================================================================
    # Topic L1 Methods
    # ========================================================================

    def create_topic_l1(
        self,
        title: str,
        descr: str,
        chunk_count: int,
        msg_count: int,
        center_vec: Optional[List[float]] = None,
        ts_from: Optional[datetime] = None,
        ts_to: Optional[datetime] = None,
        parent_l2_id: Optional[int] = None,
        vector_store: Optional[Any] = None
    ) -> int:
        """
        Create a new L1 topic and return its ID.
        
        Args:
            title: Topic title
            descr: Topic description
            chunk_count: Number of chunks in this topic
            msg_count: Number of messages in this topic
            center_vec: Center vector for the topic (stored in SQLite as JSON, optionally synced to chroma_db)
            ts_from: Start timestamp
            ts_to: End timestamp
            parent_l2_id: Parent L2 topic ID
            vector_store: VectorStore instance for saving center_vec to chroma_db (optional, for stage5)
            
        Returns:
            Topic ID
        """
        session = self.get_session()
        try:
            # Convert center_vec to JSON string for storage in SQLite
            center_vec_json = None
            if center_vec is not None:
                center_vec_list = center_vec if isinstance(center_vec, list) else center_vec.tolist() if hasattr(center_vec, 'tolist') else list(center_vec)
                center_vec_json = json.dumps(center_vec_list)
            
            # Store center_vec_json in SQLite
            topic = TopicL1Model(
                title=title,
                descr=descr,
                chunk_count=chunk_count,
                msg_count=msg_count,
                ts_from=ts_from,
                ts_to=ts_to,
                parent_l2_id=parent_l2_id,
                center_vec_json=center_vec_json
            )
            session.add(topic)
            session.commit()
            topic_id = topic.id
            
            # Optionally save center_vec to chroma_db if vector_store is provided (for stage5 sync)
            if center_vec is not None and vector_store is not None:
                try:
                    l1_topic_id = f"l1-{topic_id}"
                    center_vec_list = center_vec if isinstance(center_vec, list) else center_vec.tolist() if hasattr(center_vec, 'tolist') else list(center_vec)
                    syslog2(LOG_DEBUG, "saving l1 topic to chroma_db", 
                           topic_id=topic_id, 
                           l1_topic_id=l1_topic_id,
                           center_vec_type=type(center_vec).__name__,
                           center_vec_dim=len(center_vec_list))
                    
                    vector_store.topics_l1_collection.add(
                        ids=[l1_topic_id],
                        embeddings=[center_vec_list],
                        metadatas=[{
                            "topic_l1_id": topic_id,
                            "title": title,
                            "chunk_count": chunk_count,
                            "msg_count": msg_count
                        }]
                    )
                    syslog2(LOG_DEBUG, "l1 topic saved to chroma_db successfully", 
                           topic_id=topic_id, l1_topic_id=l1_topic_id)
                except Exception as e:
                    # Log error but don't fail the transaction
                    syslog2(LOG_WARNING, "failed to save l1 topic to chroma_db", 
                           topic_id=topic_id, 
                           l1_topic_id=f"l1-{topic_id}",
                           error=str(e),
                           center_vec_type=type(center_vec).__name__ if center_vec is not None else None)
            
            return topic_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all_topics_l1(self) -> List[TopicL1Model]:
        """Get all L1 topics."""
        session = self.get_session()
        try:
            return session.query(TopicL1Model).all()
        finally:
            session.close()

    def get_topic_l1(self, topic_id: int) -> Optional[TopicL1Model]:
        """Get a single L1 topic by ID."""
        session = self.get_session()
        try:
            return session.query(TopicL1Model).filter(TopicL1Model.id == topic_id).first()
        finally:
            session.close()

    def get_chunks_by_topic_l1(self, topic_l1_id: int) -> List[ChunkModel]:
        """Get all chunks assigned to an L1 topic."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).filter(ChunkModel.topic_l1_id == topic_l1_id).all()
        finally:
            session.close()

    def clear_topics_l1(self) -> int:
        """Delete all L1 topics and return count (stage2 - without clearing assignments)."""
        session = self.get_session()
        try:
            # Delete all L1 topics (without clearing assignments - that's stage3)
            deleted = session.query(TopicL1Model).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_topic_l1_parent(self, topic_l1_id: int, parent_l2_id: Optional[int]) -> None:
        """Update the parent L2 topic for an L1 topic."""
        session = self.get_session()
        try:
            topic = session.query(TopicL1Model).filter(TopicL1Model.id == topic_l1_id).first()
            if topic:
                topic.parent_l2_id = parent_l2_id
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_topic_l1_info(self, topic_l1_id: int, title: str, descr: str) -> None:
        """Update L1 topic title and description."""
        session = self.get_session()
        try:
            topic = session.query(TopicL1Model).filter(TopicL1Model.id == topic_l1_id).first()
            if topic:
                topic.title = title
                topic.descr = descr
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # ========================================================================
    # Topic L2 Methods
    # ========================================================================

    def create_topic_l2(
        self,
        title: str,
        descr: str,
        chunk_count: int,
        center_vec: Optional[List[float]] = None,
        vector_store: Optional[Any] = None
    ) -> int:
        """
        Create a new L2 topic and return its ID.
        
        Args:
            title: Topic title
            descr: Topic description
            chunk_count: Number of chunks in this topic
            center_vec: Center vector for the topic (stored in SQLite as JSON, optionally synced to chroma_db)
            vector_store: VectorStore instance for saving center_vec to chroma_db (optional, for stage7)
            
        Returns:
            Topic ID
        """
        session = self.get_session()
        try:
            # Convert center_vec to JSON string for storage in SQLite
            center_vec_json = None
            if center_vec is not None:
                center_vec_list = center_vec if isinstance(center_vec, list) else center_vec.tolist() if hasattr(center_vec, 'tolist') else list(center_vec)
                center_vec_json = json.dumps(center_vec_list)
            
            # Store center_vec_json in SQLite
            topic = TopicL2Model(
                title=title,
                descr=descr,
                chunk_count=chunk_count,
                center_vec_json=center_vec_json
            )
            session.add(topic)
            session.commit()
            topic_id = topic.id
            
            # Optionally save center_vec to chroma_db if vector_store is provided (for stage7 sync)
            if center_vec is not None and vector_store is not None:
                try:
                    l2_topic_id = f"l2-{topic_id}"
                    center_vec_list = center_vec if isinstance(center_vec, list) else center_vec.tolist() if hasattr(center_vec, 'tolist') else list(center_vec)
                    syslog2(LOG_DEBUG, "saving l2 topic to chroma_db", 
                           topic_id=topic_id, 
                           l2_topic_id=l2_topic_id,
                           center_vec_type=type(center_vec).__name__,
                           center_vec_dim=len(center_vec_list))
                    
                    vector_store.topics_l2_collection.add(
                        ids=[l2_topic_id],
                        embeddings=[center_vec_list],
                        metadatas=[{
                            "topic_l2_id": topic_id,
                            "title": title,
                            "chunk_count": chunk_count
                        }]
                    )
                    syslog2(LOG_DEBUG, "l2 topic saved to chroma_db successfully", 
                           topic_id=topic_id, l2_topic_id=l2_topic_id)
                except Exception as e:
                    # Log error but don't fail the transaction
                    syslog2(LOG_WARNING, "failed to save l2 topic to chroma_db", 
                           topic_id=topic_id, 
                           l2_topic_id=f"l2-{topic_id}",
                           error=str(e),
                           center_vec_type=type(center_vec).__name__ if center_vec is not None else None)
            
            return topic_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all_topics_l2(self) -> List[TopicL2Model]:
        """Get all L2 topics."""
        session = self.get_session()
        try:
            return session.query(TopicL2Model).all()
        finally:
            session.close()

    def get_topic_l2(self, topic_id: int) -> Optional[TopicL2Model]:
        """Get a single L2 topic by ID."""
        session = self.get_session()
        try:
            return session.query(TopicL2Model).filter(TopicL2Model.id == topic_id).first()
        finally:
            session.close()

    def get_chunks_by_topic_l2(self, topic_l2_id: int) -> List[ChunkModel]:
        """Get all chunks assigned to an L2 topic (via L1)."""
        session = self.get_session()
        try:
            return session.query(ChunkModel).filter(ChunkModel.topic_l2_id == topic_l2_id).all()
        finally:
            session.close()

    def update_chunks_parent_l2(self, topic_l1_id: int, topic_l2_id: Optional[int]) -> int:
        """Updates all chunks belonging to an L1 topic to have a specific L2 topic."""
        session = self.get_session()
        try:
            # We use synchronize_session=False for performance on bulk updates
            result = session.query(ChunkModel).filter(ChunkModel.topic_l1_id == topic_l1_id).update(
                {ChunkModel.topic_l2_id: topic_l2_id}, synchronize_session=False
            )
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_l1_topics_by_l2(self, topic_l2_id: int) -> List[TopicL1Model]:
        """Get all L1 topics that belong to an L2 topic."""
        session = self.get_session()
        try:
            return session.query(TopicL1Model).filter(TopicL1Model.parent_l2_id == topic_l2_id).all()
        finally:
            session.close()

    def clear_topics_l2(self) -> int:
        """Delete all L2 topics and return count (stage4 - without clearing assignments)."""
        session = self.get_session()
        try:
            # Clear L2 references in L1 topics
            session.query(TopicL1Model).update({TopicL1Model.parent_l2_id: None})
            # Delete all L2 topics (without clearing assignments in chunks - that's stage4)
            deleted = session.query(TopicL2Model).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_topic_l2_info(
        self, 
        topic_l2_id: int, 
        title: str, 
        descr: str,
        vector_store: Optional[Any] = None
    ) -> None:
        """
        Update L2 topic title and description.
        
        Args:
            topic_l2_id: Topic ID
            title: New title
            descr: New description
            vector_store: VectorStore instance for updating metadata in chroma_db
        """
        session = self.get_session()
        try:
            topic = session.query(TopicL2Model).filter(TopicL2Model.id == topic_l2_id).first()
            if topic:
                topic.title = title
                topic.descr = descr
                session.commit()
                
                # Update metadata in chroma_db if vector_store is provided
                if vector_store is not None:
                    try:
                        vector_store.topics_l2_collection.update(
                            ids=[f"l2-{topic_l2_id}"],
                            metadatas=[{
                                "topic_l2_id": topic_l2_id,
                                "title": title,
                                "chunk_count": topic.chunk_count
                            }]
                        )
                    except Exception as e:
                        # Log error but don't fail the transaction
                        syslog2(LOG_WARNING, "failed to update l2 topic metadata in chroma_db", topic_id=topic_l2_id, error=str(e))
        except Exception as e:
            session.rollback()
            raise e
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
            
            try:
                info['topics_l1'] = session.query(TopicL1Model).count()
            except Exception:
                info['topics_l1'] = 0
            
            try:
                info['topics_l2'] = session.query(TopicL2Model).count()
            except Exception:
                info['topics_l2'] = 0
            
            return info
        finally:
            session.close()
