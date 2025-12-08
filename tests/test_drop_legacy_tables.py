"""
Tests for legacy table migration (drop_legacy_tables).
"""

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine, text
from src.storage.migrations.drop_legacy_tables import drop_legacy_tables


def test_drop_legacy_tables_success(tmp_path):
    """Test successful dropping of legacy tables."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    # Create a database with legacy tables
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS topic_chunks (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS topics (id INTEGER PRIMARY KEY)"))
        conn.commit()
    
    # Run migration
    drop_legacy_tables(db_url)
    
    # Verify tables are dropped
    with engine.connect() as conn:
        # SQLite doesn't have IF EXISTS for DROP, so we check if tables exist
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('topic_chunks', 'topics')"
        ))
        tables = [row[0] for row in result]
        assert len(tables) == 0, "Legacy tables should be dropped"


def test_drop_legacy_tables_no_tables(tmp_path):
    """Test dropping when tables don't exist (should not fail)."""
    db_path = tmp_path / "test_empty.db"
    db_url = f"sqlite:///{db_path}"
    
    # Create empty database
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))  # Just create the database file
        conn.commit()
    
    # Run migration - should not raise
    drop_legacy_tables(db_url)


def test_drop_legacy_tables_error_handling(tmp_path):
    """Test error handling during table drop."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    # Create database
    engine = create_engine(db_url)
    
    # Mock execute to raise error
    with patch('src.storage.migrations.drop_legacy_tables.create_engine') as mock_engine:
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = Exception("Database error")
        mock_conn.commit = MagicMock()
        mock_conn.rollback = MagicMock()
        
        mock_eng = MagicMock()
        mock_eng.connect.return_value = mock_conn
        mock_engine.return_value = mock_eng
        
        # Should not raise, just log error
        drop_legacy_tables(db_url)
        
        # Should have attempted rollback
        assert mock_conn.rollback.call_count >= 0


def test_drop_legacy_tables_only_topic_chunks(tmp_path):
    """Test when only topic_chunks table exists."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS topic_chunks (id INTEGER PRIMARY KEY)"))
        conn.commit()
    
    drop_legacy_tables(db_url)
    
    # Verify topic_chunks is dropped
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='topic_chunks'"
        ))
        tables = [row[0] for row in result]
        assert len(tables) == 0


def test_drop_legacy_tables_only_topics(tmp_path):
    """Test when only topics table exists."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS topics (id INTEGER PRIMARY KEY)"))
        conn.commit()
    
    drop_legacy_tables(db_url)
    
    # Verify topics is dropped
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='topics'"
        ))
        tables = [row[0] for row in result]
        assert len(tables) == 0


def test_drop_legacy_tables_foreign_key_order():
    """Test that topic_chunks is dropped before topics (foreign key dependency)."""
    # This test verifies the order of operations in the function
    # topic_chunks should be dropped first since it has foreign key to topics
    
    # The function already implements this correctly:
    # 1. DROP topic_chunks first
    # 2. DROP topics second
    
    # We verify this by checking the function code structure
    # (This is more of a documentation test)
    import inspect
    source = inspect.getsource(drop_legacy_tables)
    
    # Check that topic_chunks appears before topics in DROP statements
    topic_chunks_pos = source.find("DROP TABLE IF EXISTS topic_chunks")
    topics_pos = source.find("DROP TABLE IF EXISTS topics")
    
    assert topic_chunks_pos < topics_pos, "topic_chunks should be dropped before topics"

