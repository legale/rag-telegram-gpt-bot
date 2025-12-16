"""
Tests for SqliteChunkStore.
"""

import pytest
from datetime import datetime
from src.storage.sqlite import SqliteChunkStore
from src.core.domain import Chunk, TopicUpdate
from src.storage.db import Database


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    db = Database(f"sqlite:///{db_path}")
    return db


class TestSqliteChunkStore:
    """Tests for SqliteChunkStore class."""
    
    def test_init(self, temp_db):
        """Test initialization."""
        store = SqliteChunkStore(temp_db)
        assert store.db == temp_db
        assert store._fts_index is None
    
    def test_save_batch_empty(self, temp_db):
        """Test save_batch with empty list."""
        store = SqliteChunkStore(temp_db)
        result = store.save_batch([])
        assert result == 0
    
    def test_save_batch_single_chunk(self, temp_db):
        """Test save_batch with single chunk."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(
            id="chunk1",
            text="Test chunk",
            metadata={"chat_id": "chat1"}
        )
        result = store.save_batch([chunk])
        assert result == 1
    
    def test_save_batch_multiple_chunks(self, temp_db):
        """Test save_batch with multiple chunks."""
        store = SqliteChunkStore(temp_db)
        chunks = [
            Chunk(id=f"chunk{i}", text=f"Chunk {i}", metadata={})
            for i in range(3)
        ]
        result = store.save_batch(chunks)
        assert result == 3
    
    def test_save_batch_with_embedding(self, temp_db):
        """Test save_batch with embedding."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(
            id="chunk1",
            text="Test chunk",
            embedding=[0.1] * 384
        )
        result = store.save_batch([chunk])
        assert result == 1
    
    def test_save_batch_with_msg_ids(self, temp_db):
        """Test save_batch with message IDs."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(
            id="chunk1",
            text="Test chunk",
            msg_ids=("msg1", "msg2")
        )
        result = store.save_batch([chunk])
        assert result == 1
    
    def test_save_batch_with_valid_period(self, temp_db):
        """Test save_batch with valid period."""
        store = SqliteChunkStore(temp_db)
        now = datetime.now()
        chunk = Chunk(
            id="chunk1",
            text="Test chunk",
            valid_period=(now, now)
        )
        result = store.save_batch([chunk])
        assert result == 1
    
    def test_get_fts_index(self, temp_db):
        """Test get_fts_index method."""
        store = SqliteChunkStore(temp_db)
        fts_index = store.get_fts_index()
        assert fts_index is not None
        assert store._fts_index is not None
    
    def test_get_by_ids_empty(self, temp_db):
        """Test get_by_ids with empty list."""
        store = SqliteChunkStore(temp_db)
        result = store.get_by_ids([])
        assert result == []
    
    def test_get_by_ids_single(self, temp_db):
        """Test get_by_ids with single ID."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        store.save_batch([chunk])
        
        results = store.get_by_ids(["chunk1"])
        assert len(results) == 1
        assert results[0].id == "chunk1"
    
    def test_get_by_ids_multiple(self, temp_db):
        """Test get_by_ids with multiple IDs."""
        store = SqliteChunkStore(temp_db)
        chunks = [
            Chunk(id=f"chunk{i}", text=f"Test {i}", metadata={})
            for i in range(3)
        ]
        store.save_batch(chunks)
        
        results = store.get_by_ids(["chunk0", "chunk1", "chunk2"])
        assert len(results) == 3
    
    def test_get_by_ids_nonexistent(self, temp_db):
        """Test get_by_ids with nonexistent IDs."""
        store = SqliteChunkStore(temp_db)
        results = store.get_by_ids(["nonexistent"])
        assert results == []
    
    def test_update_topics_empty(self, temp_db):
        """Test update_topics with empty dict."""
        store = SqliteChunkStore(temp_db)
        store.update_topics({})  # Should not raise
    
    def test_update_topics_single(self, temp_db):
        """Test update_topics with single update."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        store.save_batch([chunk])
        
        update = TopicUpdate(topic_ids=["l1:123"], metadata={})
        store.update_topics({"chunk1": update})  # Should not raise
    
    def test_clear(self, temp_db):
        """Test clear method."""
        store = SqliteChunkStore(temp_db)
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        store.save_batch([chunk])
        
        store.clear()  # Should not raise
        results = store.get_by_ids(["chunk1"])
        assert results == []
    

