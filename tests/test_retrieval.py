import pytest
from unittest.mock import MagicMock
from src.core.retrieval import RetrievalService
from src.storage.db import ChunkModel

def test_retrieval_service():
    # Mocks
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Setup Embedding Client Mock
    mock_embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3]
    
    # Setup Vector Store Mock
    mock_vector_store.collection.query.return_value = {
        'ids': [['1', '2']],
        'distances': [[0.1, 0.5]]
    }
    
    # Setup DB Mock
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session
    
    # Mock DB query results
    chunk1 = ChunkModel(id='1', text='Text 1')
    chunk2 = ChunkModel(id='2', text='Text 2')
    
    # Configure side_effect for filter_by to return different chunks
    # This is a bit tricky with chained calls.
    # Simplified: mock_session.query().filter_by().first()
    
    # We can mock the query object
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    
    # When filter_by is called, we need to return a mock that has .first()
    # We can use a side_effect to check args
    def filter_by_side_effect(id):
        mock_filter = MagicMock()
        if id == '1':
            mock_filter.first.return_value = chunk1
        elif id == '2':
            mock_filter.first.return_value = chunk2
        else:
            mock_filter.first.return_value = None
        return mock_filter
        
    mock_query.filter_by.side_effect = filter_by_side_effect
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    results = service.retrieve("query")
    
    assert len(results) == 2
    assert results[0]['id'] == '1'
    assert results[0]['text'] == 'Text 1'
    assert results[1]['id'] == '2'
    assert results[1]['text'] == 'Text 2'
    
    mock_embedding_client.get_embedding.assert_called_with("query")
    mock_vector_store.collection.query.assert_called_once()
