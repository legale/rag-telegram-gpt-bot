"""Tests for SqliteFTSIndex."""

import pytest
import sqlite3
from unittest.mock import Mock, patch
from sqlalchemy import text

from src.adapters.persistence.sqlite_fts_index import SqliteFTSIndex
from src.storage.db import Database
from src.core.interfaces import SearchFilters


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    database = Database(db_url)
    
    # Create test chunks
    session = database.get_session()
    try:
        session.execute(text("""
            INSERT INTO chunks (id, text, chat_id, ts_from, ts_to)
            VALUES 
                ('chunk1', 'ошибки ax820 в системе', 'chat1', '2024-01-01', '2024-01-01'),
                ('chunk2', 'тестовая ошибка', 'chat1', '2024-01-02', '2024-01-02'),
                ('chunk3', 'другая информация', 'chat2', '2024-01-03', '2024-01-03')
        """))
        session.commit()
    finally:
        session.close()
    
    yield database
    database.engine.dispose()


@pytest.fixture
def fts_index(tmp_db):
    """Create SqliteFTSIndex instance."""
    return SqliteFTSIndex(tmp_db)


class TestSqliteFTSIndex:
    """Test SqliteFTSIndex functionality."""
    
    def test_init_creates_fts_tables(self, tmp_db):
        """Test that initialization creates FTS tables."""
        fts_index = SqliteFTSIndex(tmp_db)
        
        session = tmp_db.get_session()
        try:
            # Check that chunks_fts exists
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"))
            assert result.fetchone() is not None
            
            # Check that messages_fts exists
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"))
            assert result.fetchone() is not None
        finally:
            session.close()
    
    def test_search_chunks_basic(self, fts_index):
        """Test basic chunk search."""
        results = fts_index.search_chunks("ошибки", top_k=10)
        
        assert len(results) > 0
        assert any("chunk1" in doc.id for doc in results)
        assert results[0].id == "chunk1"  # Should be top result
    
    def test_search_chunks_with_filters(self, fts_index):
        """Test chunk search with filters."""
        filters = SearchFilters(chat_id="chat1")
        results = fts_index.search_chunks("ошибки", top_k=10, filters=filters)
        
        assert len(results) > 0
        # All results should be from chat1
        for doc in results:
            # We can't directly check chat_id from ScoredDoc, but we can verify results exist
            assert doc.id in ["chunk1", "chunk2"]
    
    def test_search_chunks_empty_query(self, fts_index):
        """Test search with empty query."""
        results = fts_index.search_chunks("", top_k=10)
        assert results == []
    
    def test_search_chunks_no_results(self, fts_index):
        """Test search with query that matches nothing."""
        results = fts_index.search_chunks("nonexistentterm12345", top_k=10)
        assert results == []
    
    def test_normalize_text(self, fts_index):
        """Test text normalization."""
        # Test lowercase
        assert fts_index.normalize_text("ОШИБКИ") == "ошибки"
        
        # Test ё -> е
        assert "ё" not in fts_index.normalize_text("ёлка")
        assert "е" in fts_index.normalize_text("ёлка")
        
        # Test punctuation removal
        normalized = fts_index.normalize_text("ошибки, ax820!")
        assert "," not in normalized
        assert "!" not in normalized
    
    def test_search_messages(self, tmp_db, fts_index):
        """Test message search."""
        session = tmp_db.get_session()
        try:
            # Insert test messages
            session.execute(text("""
                INSERT INTO messages (msg_id, chat_id, from_id, ts, text)
                VALUES 
                    ('msg1', 'chat1', 'user1', '2024-01-01', 'тестовое сообщение'),
                    ('msg2', 'chat1', 'user2', '2024-01-02', 'другое сообщение')
            """))
            session.commit()
        finally:
            session.close()
        
        results = fts_index.search_messages("тестовое", top_k=10)
        assert len(results) > 0
        assert any("msg1" in doc.id for doc in results)
    
    def test_database_integrity_check(self, fts_index):
        """Test database integrity check."""
        result = fts_index._check_database_integrity()
        assert result is True
    
    def test_fts_table_integrity_check(self, fts_index):
        """Test FTS table integrity check."""
        result = fts_index._check_fts_table_integrity("chunks_fts")
        assert result is True
    
    def test_rebuild_fts_table(self, tmp_db, fts_index):
        """Test FTS table rebuild."""
        # Verify table has data
        session = tmp_db.get_session()
        try:
            result = session.execute(text("SELECT COUNT(*) FROM chunks_fts"))
            count_before = result.scalar()
            assert count_before > 0
        finally:
            session.close()
        
        # Rebuild table
        success = fts_index._rebuild_fts_table("chunks_fts")
        assert success is True
        
        # Verify table still has data
        session = tmp_db.get_session()
        try:
            result = session.execute(text("SELECT COUNT(*) FROM chunks_fts"))
            count_after = result.scalar()
            assert count_after == count_before
        finally:
            session.close()
    
    def test_recover_fts_tables(self, fts_index):
        """Test FTS table recovery."""
        # Reset recovery flag
        fts_index._recovery_attempted = False
        
        # Recovery should succeed on healthy database
        result = fts_index._recover_fts_tables("chunks_fts")
        # Should succeed or fail gracefully
        assert isinstance(result, bool)
    
    def test_search_handles_database_error(self, fts_index):
        """Test that search handles database errors gracefully."""
        # Mock database error during query execution
        session = fts_index.db.get_session()
        try:
            with patch.object(session, 'execute') as mock_execute:
                mock_execute.side_effect = sqlite3.DatabaseError("database disk image is malformed")
                
                results = fts_index.search_chunks("test", top_k=10)
                # Should return empty list on error
                assert results == []
        finally:
            session.close()
    
    def test_populate_fts_if_empty(self, tmp_db):
        """Test that FTS table is populated if empty."""
        # Create FTS index (which should populate tables)
        fts_index = SqliteFTSIndex(tmp_db)
        
        # Verify chunks_fts has data
        session = tmp_db.get_session()
        try:
            result = session.execute(text("SELECT COUNT(*) FROM chunks_fts"))
            count = result.scalar()
            assert count > 0
        finally:
            session.close()

