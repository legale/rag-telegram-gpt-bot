"""
Tests for message search functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.core.message_search import search_message_links
from src.core.retrieval import RetrievalService
from src.storage.db import Database


@pytest.fixture
def mock_retrieval():
    """Create a mock RetrievalService."""
    retrieval = MagicMock(spec=RetrievalService)
    return retrieval


@pytest.fixture
def mock_db():
    """Create a mock Database."""
    db = MagicMock(spec=Database)
    return db


def test_search_message_links_success(mock_retrieval, mock_db):
    """Test successful message link search."""
    # Mock retrieval results
    mock_retrieval.search_chunks_basic.return_value = [
        {"id": "chunk1", "score": 0.9},
        {"id": "chunk2", "score": 0.8},
        {"id": "chunk3", "score": 0.7},
    ]
    
    # Mock database link info
    mock_db.get_chunk_link_info.side_effect = [
        (123456, 100, "test_channel"),  # chunk1
        (-987654, 200, None),  # chunk2 (group, negative chat_id)
        (111222, 300, "another_channel"),  # chunk3
    ]
    
    links = search_message_links(mock_retrieval, mock_db, "test query", top_k=3)
    
    assert len(links) == 3
    assert "t.me/test_channel/100" in links[0]
    # Negative chat_id without 100 prefix uses tg:// format
    assert "tg://" in links[1] and "987654" in links[1]
    assert "t.me/another_channel/300" in links[2]
    
    mock_retrieval.search_chunks_basic.assert_called_once_with("test query", n_results=3)
    assert mock_db.get_chunk_link_info.call_count == 3


def test_search_message_links_empty_results(mock_retrieval, mock_db):
    """Test search with no results."""
    mock_retrieval.search_chunks_basic.return_value = []
    
    links = search_message_links(mock_retrieval, mock_db, "nonexistent query", top_k=3)
    
    assert len(links) == 0
    mock_retrieval.search_chunks_basic.assert_called_once_with("nonexistent query", n_results=3)
    mock_db.get_chunk_link_info.assert_not_called()


def test_search_message_links_missing_chunk_id(mock_retrieval, mock_db):
    """Test search results without chunk IDs are skipped."""
    mock_retrieval.search_chunks_basic.return_value = [
        {"score": 0.9},  # Missing 'id'
        {"id": "chunk2", "score": 0.8},
    ]
    
    mock_db.get_chunk_link_info.return_value = (123456, 100, "test_channel")
    
    links = search_message_links(mock_retrieval, mock_db, "query", top_k=3)
    
    # Only chunk2 should be processed
    assert len(links) == 1
    mock_db.get_chunk_link_info.assert_called_once_with("chunk2")


def test_search_message_links_missing_link_info(mock_retrieval, mock_db):
    """Test chunks with missing link info are skipped."""
    mock_retrieval.search_chunks_basic.return_value = [
        {"id": "chunk1", "score": 0.9},
        {"id": "chunk2", "score": 0.8},
    ]
    
    # First chunk has missing info, second has valid info
    mock_db.get_chunk_link_info.side_effect = [
        (None, None, None),  # chunk1 - missing info
        (123456, 100, "test_channel"),  # chunk2 - valid info
    ]
    
    links = search_message_links(mock_retrieval, mock_db, "query", top_k=3)
    
    # Only chunk2 should be included
    assert len(links) == 1
    assert "t.me/test_channel/100" in links[0]


def test_search_message_links_default_top_k(mock_retrieval, mock_db):
    """Test default top_k value."""
    mock_retrieval.search_chunks_basic.return_value = []
    
    search_message_links(mock_retrieval, mock_db, "query")
    
    # Should use default top_k=3
    mock_retrieval.search_chunks_basic.assert_called_once_with("query", n_results=3)


def test_search_message_links_custom_top_k(mock_retrieval, mock_db):
    """Test custom top_k value."""
    mock_retrieval.search_chunks_basic.return_value = []
    
    search_message_links(mock_retrieval, mock_db, "query", top_k=5)
    
    mock_retrieval.search_chunks_basic.assert_called_once_with("query", n_results=5)


def test_search_message_links_negative_chat_id_with_prefix(mock_retrieval, mock_db):
    """Test handling of negative chat_id with '100' prefix."""
    mock_retrieval.search_chunks_basic.return_value = [
        {"id": "chunk1", "score": 0.9},
    ]
    
    # Negative chat_id like -1001234567890 (Telegram format)
    # Should remove '100' prefix: 1001234567890 -> 1234567890
    mock_db.get_chunk_link_info.return_value = (-1001234567890, 100, None)
    
    links = search_message_links(mock_retrieval, mock_db, "query", top_k=1)
    
    assert len(links) == 1
    # Should convert -1001234567890 -> 1234567890
    assert "t.me/c/1234567890/100" in links[0]

