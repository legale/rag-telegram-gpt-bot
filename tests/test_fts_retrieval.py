"""Tests for FTS-only retrieval."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.core.hybrid_retrieval import HybridRetrievalService
from src.core.domain import Chunk, Message
from src.core.interfaces import ScoredDoc, SearchFilters


@pytest.fixture
def mock_fts_index():
    """Create mock FTS index."""
    mock = Mock()
    mock.search_chunks.return_value = [
        ScoredDoc(id="chunk1", score=10.0, meta={}),
        ScoredDoc(id="chunk2", score=8.0, meta={}),
    ]
    return mock


@pytest.fixture
def mock_vector_index():
    """Create mock vector index."""
    return Mock()


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    mock = Mock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_chunk_store():
    """Create mock chunk store."""
    mock = Mock()
    chunk1 = Chunk(
        id="chunk1",
        text="ошибки ax820 в системе",
        msg_ids=("msg1", "msg2"),
        valid_period=(datetime(2024, 1, 1), datetime(2024, 1, 1)),
        metadata={"chat_id": "chat1"},
        embedding=[0.1, 0.2, 0.3]
    )
    chunk2 = Chunk(
        id="chunk2",
        text="тестовая ошибка",
        msg_ids=("msg3", "msg3"),
        valid_period=(datetime(2024, 1, 2), datetime(2024, 1, 2)),
        metadata={"chat_id": "chat1"},
        embedding=[0.2, 0.3, 0.4]
    )
    mock.get_by_ids.return_value = [chunk1, chunk2]
    return mock


@pytest.fixture
def mock_message_store():
    """Create mock message store."""
    mock = Mock()
    mock.get_context.return_value = []  # Return empty list by default
    return mock


@pytest.fixture
def hybrid_service(mock_fts_index, mock_vector_index, mock_embedder, mock_chunk_store, mock_message_store):
    """Create HybridRetrievalService instance."""
    return HybridRetrievalService(
        fts_index=mock_fts_index,
        vector_index=mock_vector_index,
        embedder=mock_embedder,
        chunk_store=mock_chunk_store,
        message_store=mock_message_store,
        log_level=7,  # LOG_DEBUG
        fts_only=False
    )


@pytest.fixture
def fts_only_service(mock_fts_index, mock_vector_index, mock_embedder, mock_chunk_store, mock_message_store):
    """Create HybridRetrievalService instance with FTS-only mode."""
    return HybridRetrievalService(
        fts_index=mock_fts_index,
        vector_index=mock_vector_index,
        embedder=mock_embedder,
        chunk_store=mock_chunk_store,
        message_store=mock_message_store,
        log_level=7,  # LOG_DEBUG
        fts_only=True
    )


class TestFTSOnlyRetrieval:
    """Test FTS-only retrieval mode."""
    
    def test_fts_only_skips_vector_reranking(self, fts_only_service, mock_embedder):
        """Test that FTS-only mode skips vector reranking."""
        results = fts_only_service.search("ошибки ax820", top_k=10, rerank_top_k=5)
        
        # Embedder should not be called in FTS-only mode
        mock_embedder.embed_query.assert_not_called()
        
        # Should return results based on FTS scores only
        assert len(results) > 0
    
    def test_fts_only_returns_fts_results(self, fts_only_service):
        """Test that FTS-only mode returns FTS results."""
        results = fts_only_service.search("ошибки", top_k=10, rerank_top_k=5)
        
        assert len(results) > 0
        # Results should be sorted by FTS score
        assert results[0].chunk.id == "chunk1"  # Highest FTS score
    
    def test_hybrid_mode_uses_vector_reranking(self, hybrid_service, mock_embedder):
        """Test that hybrid mode uses vector reranking."""
        results = hybrid_service.search("ошибки", top_k=10, rerank_top_k=5)
        
        # Embedder should be called in hybrid mode
        mock_embedder.embed_query.assert_called_once()
        
        # Should return results
        assert len(results) > 0
    
    def test_fts_only_handles_empty_results(self, fts_only_service, mock_fts_index):
        """Test FTS-only mode handles empty FTS results."""
        mock_fts_index.search_chunks.return_value = []
        
        results = fts_only_service.search("nonexistent", top_k=10)
        
        assert results == []
    
    def test_fts_only_handles_fts_error(self, fts_only_service, mock_fts_index):
        """Test FTS-only mode handles FTS search errors."""
        mock_fts_index.search_chunks.side_effect = Exception("FTS error")
        
        results = fts_only_service.search("test", top_k=10)
        
        assert results == []
    
    def test_hybrid_fallback_on_embedding_error(self, hybrid_service, mock_embedder):
        """Test hybrid mode falls back to FTS-only if embedding fails."""
        mock_embedder.embed_query.side_effect = Exception("Embedding error")
        
        results = hybrid_service.search("ошибки", top_k=10, rerank_top_k=5)
        
        # Should still return results using FTS scores only
        assert len(results) > 0


class TestFTSRetrievalWithFilters:
    """Test FTS retrieval with filters."""
    
    def test_search_with_chat_filter(self, fts_only_service, mock_fts_index):
        """Test search with chat_id filter."""
        filters = SearchFilters(chat_id="chat1")
        
        results = fts_only_service.search("ошибки", top_k=10, filters=filters)
        
        # Verify filter was passed to FTS index
        mock_fts_index.search_chunks.assert_called_once()
        call_args = mock_fts_index.search_chunks.call_args
        assert call_args.kwargs.get("filters") == filters
    
    def test_search_with_time_filter(self, fts_only_service, mock_fts_index):
        """Test search with time range filter."""
        filters = SearchFilters(
            time_from=datetime(2024, 1, 1),
            time_to=datetime(2024, 1, 2)
        )
        
        results = fts_only_service.search("ошибки", top_k=10, filters=filters)
        
        # Verify filter was passed
        call_args = mock_fts_index.search_chunks.call_args
        assert call_args.kwargs.get("filters") == filters


class TestFTSRetrievalCompatibility:
    """Test compatibility methods for FTS retrieval."""
    
    def test_retrieve_method(self, fts_only_service):
        """Test retrieve() compatibility method."""
        results = fts_only_service.retrieve("ошибки", n_results=5)
        
        assert isinstance(results, list)
        # Results should be in dict format
        if results:
            assert "id" in results[0]
            assert "text" in results[0]
    
    def test_search_chunks_basic(self, fts_only_service):
        """Test search_chunks_basic() compatibility method."""
        results = fts_only_service.search_chunks_basic("ошибки", n_results=3)
        
        assert isinstance(results, list)
        # Results should have distance field
        if results:
            assert "distance" in results[0]
            assert "id" in results[0]

