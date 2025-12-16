"""SQLite implementation of ChunkStore interface."""

from __future__ import annotations

from typing import Dict, List, Optional
import json

from src.core.domain import Chunk, TopicUpdate
from src.core.interfaces import ChunkStore, FTSIndex
from src.storage.db import Database, ChunkModel
from .sqlite_fts_index import SqliteFTSIndex


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

