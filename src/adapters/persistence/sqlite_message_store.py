"""SQLite implementation of MessageStore interface."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional
import json

from src.core.domain import Message
from src.core.interfaces import MessageStore, FTSIndex
from src.storage.db import Database, MessageModel, MessageMetaModel
from .sqlite_fts_index import SqliteFTSIndex


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
        session = self.db.get_session()
        try:
            # Query with limit and offset
            query = session.query(MessageModel).filter(
                MessageModel.chat_id == chat_id
            ).order_by(MessageModel.ts).offset(offset).limit(limit)
            
            models = query.all()
            return [self._model_to_domain(model, session) for model in models]
        finally:
            session.close()

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
            time_start = time_point - timedelta(seconds=window_sec)
            time_end = time_point + timedelta(seconds=window_sec)
            
            query = session.query(MessageModel).filter(
                MessageModel.chat_id == chat_id,
                MessageModel.ts >= time_start,
                MessageModel.ts <= time_end
            ).order_by(MessageModel.ts)
            
            models = query.all()
            return [self._model_to_domain(model, session) for model in models]
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

