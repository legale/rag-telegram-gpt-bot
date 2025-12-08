from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from typing import List, Optional, Tuple
import json

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
    ts_from = Column(DateTime, nullable=True, index=True)
    ts_to = Column(DateTime, nullable=True)
    
    # Topic assignments
    topic_l1_id = Column(Integer, ForeignKey('topics_l1.id', ondelete='SET NULL'), nullable=True, index=True)
    topic_l2_id = Column(Integer, ForeignKey('topics_l2.id', ondelete='SET NULL'), nullable=True, index=True)

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
    center_vec = Column(Text, nullable=True)  # JSON-serialized vector
    ts_from = Column(DateTime, nullable=True)
    ts_to = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    center_vec = Column(Text, nullable=True)  # JSON-serialized vector
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chunks = relationship("ChunkModel", back_populates="topic_l2")
    topics_l1 = relationship("TopicL1Model", back_populates="parent_l2")


# Legacy models (kept for backward compatibility)
class TopicModel(Base):
    """Legacy topic model (from old simple clustering)."""
    __tablename__ = 'topics'
    
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TopicChunkModel(Base):
    """Legacy topic-chunk mapping (from old simple clustering)."""
    __tablename__ = 'topic_chunks'
    
    topic_id = Column(Integer, ForeignKey('topics.id', ondelete='CASCADE'), primary_key=True)
    chunk_id = Column(String, ForeignKey('chunks.id', ondelete='CASCADE'), primary_key=True)


# ============================================================================
# Database Class
# ============================================================================

class Database:
    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("db_url must be provided")
        self.db_url = db_url
        self.engine = create_engine(db_url)
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
                    print(f"Schema update warning (chunks 14.2): {e}")

            # Check chunks table for topic_l1_id (Phase 14.3)
            try:
                conn.execute(text("SELECT topic_l1_id FROM chunks LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN topic_l1_id INTEGER"))
                    conn.execute(text("ALTER TABLE chunks ADD COLUMN topic_l2_id INTEGER"))
                    conn.commit()
                except Exception as e:
                    print(f"Schema update warning (chunks 14.3): {e}")

            # Check topics_l1 table for parent_l2_id
            try:
                conn.execute(text("SELECT parent_l2_id FROM topics_l1 LIMIT 1"))
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE topics_l1 ADD COLUMN parent_l2_id INTEGER"))
                    conn.commit()
                except Exception as e:
                     print(f"Schema update warning (topics_l1): {e}")
        
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

    def add_messages_batch(self, messages: List[dict]) -> None:
        """
        Add multiple messages to the database in one transaction.
        Args:
            messages: List of dictionaries matching MessageModel fields.
        """
        session = self.get_session()
        try:
            # Use bulk_insert_mappings for even better performance if needed, 
            # but add_all is fine for now with ORM objects.
            # To avoid creating objects manually, let's accept dicts and map them.
            models = [MessageModel(**msg) for msg in messages]
            session.add_all(models)
            session.commit()
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
        parent_l2_id: Optional[int] = None
    ) -> int:
        """Create a new L1 topic and return its ID."""
        session = self.get_session()
        try:
            center_vec_json = json.dumps(center_vec) if center_vec else None
            topic = TopicL1Model(
                title=title,
                descr=descr,
                chunk_count=chunk_count,
                msg_count=msg_count,
                center_vec=center_vec_json,
                ts_from=ts_from,
                ts_to=ts_to,
                parent_l2_id=parent_l2_id
            )
            session.add(topic)
            session.commit()
            return topic.id
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
        """Delete all L1 topics and return count."""
        session = self.get_session()
        try:
            # Clear topic assignments in chunks first
            session.query(ChunkModel).update({ChunkModel.topic_l1_id: None})
            # Delete all L1 topics
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
        center_vec: Optional[List[float]] = None
    ) -> int:
        """Create a new L2 topic and return its ID."""
        session = self.get_session()
        try:
            center_vec_json = json.dumps(center_vec) if center_vec else None
            topic = TopicL2Model(
                title=title,
                descr=descr,
                chunk_count=chunk_count,
                center_vec=center_vec_json
            )
            session.add(topic)
            session.commit()
            return topic.id
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
        """Delete all L2 topics and return count."""
        session = self.get_session()
        try:
            # Clear L2 references in L1 topics
            session.query(TopicL1Model).update({TopicL1Model.parent_l2_id: None})
            # Clear L2 assignments in chunks
            session.query(ChunkModel).update({ChunkModel.topic_l2_id: None})
            # Delete all L2 topics
            deleted = session.query(TopicL2Model).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_topic_l2_info(self, topic_l2_id: int, title: str, descr: str) -> None:
        """Update L2 topic title and description."""
        session = self.get_session()
        try:
            topic = session.query(TopicL2Model).filter(TopicL2Model.id == topic_l2_id).first()
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
    # Legacy Topic Methods (for backward compatibility)
    # ========================================================================

    def create_topic(self, title: str, description: str) -> int:
        """Creates a new legacy topic and returns its ID."""
        session = self.get_session()
        try:
            topic = TopicModel(title=title, description=description)
            session.add(topic)
            session.commit()
            return topic.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_topic_chunks(self, topic_id: int, chunk_ids: list[str]) -> int:
        """Associates multiple chunks with a legacy topic."""
        if not chunk_ids:
            return 0
        session = self.get_session()
        try:
            mappings = [
                TopicChunkModel(topic_id=topic_id, chunk_id=cid)
                for cid in chunk_ids
            ]
            session.add_all(mappings)
            session.commit()
            return len(mappings)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_topics(self) -> tuple[int, int]:
        """
        Deletes all legacy topics (and cascades to topic_chunks).
        Returns (num_topics_deleted, num_mappings_deleted).
        """
        session = self.get_session()
        try:
            num_topics = session.query(TopicModel).count()
            if num_topics == 0:
                return 0, 0
                
            session.rollback()
            
            num_mappings = session.query(TopicChunkModel).delete()
            num_topics = session.query(TopicModel).delete()
            
            session.commit()
            return num_topics, num_mappings
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all_topics(self):
        """Returns all legacy topics."""
        session = self.get_session()
        try:
            return session.query(TopicModel).all()
        finally:
            session.close()

    def get_topic(self, topic_id: int):
        """Returns a single legacy topic by ID."""
        session = self.get_session()
        try:
            return session.query(TopicModel).filter(TopicModel.id == topic_id).first()
        finally:
            session.close()

    def get_topic_chunks(self, topic_id: int):
        """Returns all chunk IDs for a legacy topic."""
        session = self.get_session()
        try:
            results = session.query(TopicChunkModel.chunk_id).filter(TopicChunkModel.topic_id == topic_id).all()
            return [r[0] for r in results]
        finally:
            session.close()
