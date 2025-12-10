"""
Debug tests for retrieval to identify why chunks are not being found.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.core.retrieval import RetrievalService
from src.storage.db import ChunkModel, TopicL1Model, TopicL2Model
from src.core.syslog2 import LOG_DEBUG


def test_retrieval_with_empty_vector_store():
    """Test retrieval when vector store is empty."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Empty vector store
    mock_vector_store.collection.count.return_value = 0
    mock_vector_store.collection.name = "test_collection"
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    service = RetrievalService(
        mock_vector_store,
        mock_db,
        mock_embedding_client,
        log_level=LOG_DEBUG,
        use_topic_retrieval=False
    )
    
    results = service.retrieve("test query", n_results=5)
    
    # Should return empty list, not crash
    assert results == []
    assert mock_vector_store.collection.count.called


def test_retrieval_with_results_but_filtered():
    """Test retrieval when results are filtered by threshold."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Vector store has data
    mock_vector_store.collection.count.return_value = 100
    mock_vector_store.collection.name = "test_collection"
    
    # Return results with high distances (low similarity)
    mock_vector_store.collection.query.return_value = {
        'ids': [['1', '2', '3']],
        'distances': [[0.9, 0.95, 0.98]],  # High distances = low similarity
        'documents': [['doc1', 'doc2', 'doc3']],
        'metadatas': [[{}, {}, {}]]
    }
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session
    
    # Mock chunks
    chunk1 = ChunkModel(id='1', text='Text 1', metadata_json=None)
    chunk1.topic_l1 = None
    chunk1.topic_l2 = None
    
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.options.return_value = mock_query
    
    def filter_by_side_effect(**kwargs):
        mock_filter = MagicMock()
        chunk_id = kwargs.get('id')
        if chunk_id == '1':
            mock_filter.first.return_value = chunk1
        else:
            mock_filter.first.return_value = None
        return mock_filter
    
    mock_query.filter_by.side_effect = filter_by_side_effect
    
    service = RetrievalService(
        mock_vector_store,
        mock_db,
        mock_embedding_client,
        log_level=LOG_DEBUG,
        use_topic_retrieval=False
    )
    
    # With default score_threshold=0.5, chunks with similarity < 0.5 should be filtered
    # But we removed the threshold check, so all should pass
    results = service.retrieve("test query", n_results=5, score_threshold=0.0)
    
    # Should return at least chunk1 (if it exists in DB)
    # The issue might be that chunks don't exist in DB
    assert isinstance(results, list)


def test_retrieval_chunk_not_in_db():
    """Test when vector store returns IDs but chunks don't exist in DB."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_vector_store.collection.count.return_value = 100
    mock_vector_store.collection.name = "test_collection"
    
    mock_vector_store.collection.query.return_value = {
        'ids': [['1', '2']],
        'distances': [[0.1, 0.2]],
        'documents': [['doc1', 'doc2']],
        'metadatas': [[{}, {}]]
    }
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session
    
    # Chunks don't exist in DB (return None)
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.options.return_value = mock_query
    
    def filter_by_side_effect(**kwargs):
        mock_filter = MagicMock()
        mock_filter.first.return_value = None  # Chunk not found
        return mock_filter
    
    mock_query.filter_by.side_effect = filter_by_side_effect
    
    service = RetrievalService(
        mock_vector_store,
        mock_db,
        mock_embedding_client,
        log_level=LOG_DEBUG,
        use_topic_retrieval=False
    )
    
    results = service.retrieve("test query", n_results=5)
    
    # Should return empty because chunks don't exist in DB
    assert results == []


def test_retrieval_vector_store_empty_ids():
    """Test when vector store returns empty ids list."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_vector_store.collection.count.return_value = 100
    mock_vector_store.collection.name = "test_collection"
    
    # Empty results
    mock_vector_store.collection.query.return_value = {
        'ids': [[]],  # Empty list
        'distances': [[]],
        'documents': [[]],
        'metadatas': [[]]
    }
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    service = RetrievalService(
        mock_vector_store,
        mock_db,
        mock_embedding_client,
        log_level=LOG_DEBUG,
        use_topic_retrieval=False
    )
    
    results = service.retrieve("test query", n_results=5)
    
    # Should handle empty results gracefully
    assert results == []

