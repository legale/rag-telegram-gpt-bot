"""
Tests for SqliteMessageStore.
"""

import pytest
from datetime import datetime
from src.storage.sqlite import SqliteMessageStore
from src.core.domain import Message
from src.storage.db import Database


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    db = Database(f"sqlite:///{db_path}")
    return db


class TestSqliteMessageStore:
    """Tests for SqliteMessageStore class."""
    
    def test_init(self, temp_db):
        """Test initialization."""
        store = SqliteMessageStore(temp_db)
        assert store.db == temp_db
        assert store._fts_index is None
    
    def test_save_batch_empty(self, temp_db):
        """Test save_batch with empty list."""
        store = SqliteMessageStore(temp_db)
        result = store.save_batch([])
        assert result == 0
    
    def test_save_batch_single_message(self, temp_db):
        """Test save_batch with single message."""
        store = SqliteMessageStore(temp_db)
        message = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Test message",
            timestamp=datetime.now()
        )
        result = store.save_batch([message])
        assert result == 1
    
    def test_save_batch_multiple_messages(self, temp_db):
        """Test save_batch with multiple messages."""
        store = SqliteMessageStore(temp_db)
        messages = [
            Message(
                id=f"msg{i}",
                chat_id="chat1",
                from_id="user1",
                text=f"Message {i}",
                timestamp=datetime.now()
            )
            for i in range(3)
        ]
        result = store.save_batch(messages)
        assert result == 3
    
    def test_save_batch_with_metadata(self, temp_db):
        """Test save_batch with message metadata."""
        store = SqliteMessageStore(temp_db)
        message = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Test message",
            timestamp=datetime.now(),
            meta={"key": "value"}
        )
        result = store.save_batch([message])
        assert result == 1
    
    def test_get_fts_index(self, temp_db):
        """Test get_fts_index method."""
        store = SqliteMessageStore(temp_db)
        fts_index = store.get_fts_index()
        assert fts_index is not None
        assert store._fts_index is not None
    
    def test_get_by_chat(self, temp_db):
        """Test get_by_chat method."""
        store = SqliteMessageStore(temp_db)
        message = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Test message",
            timestamp=datetime.now()
        )
        store.save_batch([message])
        
        results = store.get_by_chat("chat1", limit=10, offset=0)
        assert len(results) == 1
        assert results[0].id == "msg1"
    
    def test_get_context(self, temp_db):
        """Test get_context method."""
        store = SqliteMessageStore(temp_db)
        now = datetime.now()
        message = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Test message",
            timestamp=now
        )
        store.save_batch([message])
        
        results = store.get_context("chat1", now, window_sec=60)
        assert len(results) >= 1
    
    def test_count(self, temp_db):
        """Test count method."""
        store = SqliteMessageStore(temp_db)
        initial_count = store.count()
        
        message = Message(
            id="msg1",
            chat_id="chat1",
            from_id="user1",
            text="Test message",
            timestamp=datetime.now()
        )
        store.save_batch([message])
        
        assert store.count() == initial_count + 1

