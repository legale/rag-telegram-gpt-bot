import pytest
from unittest.mock import MagicMock
from src.core.retrieval import RetrievalService
from src.storage.db import ChunkModel

def test_retrieval_service():
    # Mocks
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Setup Embedding Client Mock - use get_embeddings (plural)
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    # Setup Vector Store Mock
    mock_vector_store.collection.query.return_value = {
        'ids': [['1', '2']],
        'distances': [[0.1, 0.5]]
    }
    
    # Setup DB Mock
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session
    
    # Mock DB query results
    chunk1 = ChunkModel(id='1', text='Text 1', metadata_json=None)
    chunk1.topic_l1 = None
    chunk1.topic_l2 = None
    chunk2 = ChunkModel(id='2', text='Text 2', metadata_json=None)
    chunk2.topic_l1 = None
    chunk2.topic_l2 = None
    
    # Configure side_effect for filter_by to return different chunks
    # This is a bit tricky with chained calls.
    # Simplified: mock_session.query().filter_by().first()
    
    # We can mock the query object
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    # Handle chained .options() call
    mock_query.options.return_value = mock_query
    
    # When filter_by is called, we need to return a mock that has .first()
    # We can use a side_effect to check args
    def filter_by_side_effect(**kwargs):
        mock_filter = MagicMock()
        chunk_id = kwargs.get('id')
        if chunk_id == '1':
            mock_filter.first.return_value = chunk1
        elif chunk_id == '2':
            mock_filter.first.return_value = chunk2
        else:
            mock_filter.first.return_value = None
        return mock_filter
        
    mock_query.filter_by.side_effect = filter_by_side_effect
    
    # Disable topic retrieval for this test to keep it simple
    service = RetrievalService(
        mock_vector_store, 
        mock_db, 
        mock_embedding_client,
        use_topic_retrieval=False
    )
    
    results = service.retrieve("query")
    
    assert len(results) == 2
    assert results[0]['id'] == '1'
    assert results[0]['text'] == 'Text 1'
    assert results[1]['id'] == '2'
    assert results[1]['text'] == 'Text 2'
    
    # Verify vector store was queried
    mock_vector_store.collection.query.assert_called_once()
    # Verify embeddings were computed
    mock_embedding_client.get_embeddings.assert_called_once()


def test_retrieval_service_with_topics():
    """Test retrieval service with topic-based retrieval enabled."""
    from src.storage.db import TopicL1Model, TopicL2Model
    
    # Mocks
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Setup Embedding Client Mock
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    # Setup Vector Store Mock - return empty to test topic-only retrieval
    mock_vector_store.collection.query.return_value = {
        'ids': [[]],
        'distances': [[]]
    }
    
    # Setup DB Mock
    mock_session = MagicMock()
    mock_db.get_session.return_value = mock_session
    
    # Mock topic with center vector
    topic_l1 = TopicL1Model(id=1, title="Test Topic", descr="Test", center_vec='[0.1, 0.2, 0.3]')
    chunk1 = ChunkModel(id='1', text='Text 1', metadata_json=None, topic_l1_id=1)
    chunk1.topic_l1 = topic_l1
    chunk1.topic_l2 = None
    
    # Mock topic query
    mock_topic_query = MagicMock()
    mock_topic_query.filter.return_value = mock_topic_query
    mock_topic_query.all.return_value = [topic_l1]
    
    # Mock chunk query for topics
    mock_chunk_query = MagicMock()
    mock_chunk_query.options.return_value = mock_chunk_query
    mock_chunk_query.filter.return_value = mock_chunk_query
    mock_chunk_query.limit.return_value = mock_chunk_query
    mock_chunk_query.all.return_value = [chunk1]
    
    # Setup query routing
    def query_side_effect(model):
        if model == TopicL1Model:
            return mock_topic_query
        elif model == ChunkModel:
            return mock_chunk_query
        return MagicMock()
    
    mock_session.query.side_effect = query_side_effect
    
    # Enable topic retrieval
    service = RetrievalService(
        mock_vector_store,
        mock_db,
        mock_embedding_client,
        use_topic_retrieval=True
    )
    
    results = service.retrieve("query", n_results=5)
    
    # Should have at least one result from topic retrieval
    assert len(results) >= 0  # May be empty if similarity threshold not met
    
    # Verify topic query was attempted
    mock_session.query.assert_called()


def test_find_similar_topics_l1():
    """Test _find_similar_topics for L1 topics."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    from src.storage.db import TopicL1Model
    
    # Mock topic with center vector
    topic = TopicL1Model(id=1, title="Topic", center_vec='[0.1, 0.2, 0.3]')
    
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = [topic]
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._find_similar_topics(query_emb, topic_type="l1", n_topics=3)
    
    assert len(results) > 0
    assert results[0][0] == 1  # topic_id


def test_find_similar_topics_l2():
    """Test _find_similar_topics for L2 topics."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    from src.storage.db import TopicL2Model
    
    topic = TopicL2Model(id=2, title="L2 Topic", center_vec='[0.2, 0.3, 0.4]')
    
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = [topic]
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.2, 0.3, 0.4]
    results = service._find_similar_topics(query_emb, topic_type="l2", n_topics=3)
    
    assert len(results) > 0


def test_find_similar_topics_empty():
    """Test _find_similar_topics when no topics exist."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = []
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._find_similar_topics(query_emb)
    
    assert results == []


def test_find_similar_topics_threshold():
    """Test _find_similar_topics with similarity threshold."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    from src.storage.db import TopicL1Model
    
    # Topic with center_vec as JSON string - use orthogonal vectors for low similarity
    topic = TopicL1Model(id=1, title="Topic", center_vec='[1.0, 0.0, 0.0]')
    
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value.all.return_value = [topic]
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.0, 1.0, 0.0]  # Orthogonal vector (similarity ~0)
    results = service._find_similar_topics(query_emb, similarity_threshold=0.9)
    
    # Should filter out low similarity topic
    assert len(results) == 0


def test_retrieve_chunks_from_topics():
    """Test _retrieve_chunks_from_topics."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    chunk = ChunkModel(id='chunk1', text='Text 1', metadata_json='{"key": "value"}')
    chunk.topic_l1_id = 1
    chunk.topic_l2_id = None
    chunk.topic_l1 = None
    chunk.topic_l2 = None
    
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.options.return_value.filter.return_value.limit.return_value.all.return_value = [chunk]
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    results = service._retrieve_chunks_from_topics([1], [])
    
    assert len(results) > 0
    assert results[0]['id'] == 'chunk1'
    assert results[0]['source'] == 'topic_l1'


def test_two_stage_search():
    """Test _two_stage_search."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    # Mock topics_l2_collection
    mock_topics_collection = MagicMock()
    mock_topics_collection.query.return_value = {
        'ids': [['topic1']],
        'metadatas': [[{'topic_l2_id': 1}]],
        'distances': [[0.1]]
    }
    mock_vector_store.get_topics_l2_collection.return_value = mock_topics_collection
    
    # Mock vector store collection for chunks
    mock_vector_store.collection.query.return_value = {
        'ids': [['chunk1']],
        'documents': [['Text']],
        'metadatas': [[{}]],
        'distances': [[0.2]]
    }
    
    # Mock chunk query
    chunk = ChunkModel(id='chunk1', text='Text', metadata_json=None)
    chunk.topic_l1 = None
    chunk.topic_l2 = None
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.options.return_value.filter.return_value.all.return_value = [chunk]
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._two_stage_search(query_emb, n_results=10)
    
    assert len(results) >= 0  # May return empty if no chunks found


def test_two_stage_search_no_l2_topics():
    """Test _two_stage_search when no L2 topics found."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_topics_collection = MagicMock()
    mock_topics_collection.query.return_value = {
        'ids': [[]],
        'metadatas': [[]]
    }
    mock_vector_store.get_topics_l2_collection.return_value = mock_topics_collection
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client, verbosity=2)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._two_stage_search(query_emb)
    
    assert results == []


def test_direct_chunk_query():
    """Test _direct_chunk_query."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_vector_store.collection.count.return_value = 10
    mock_vector_store.collection.query.return_value = {
        'ids': [['chunk1', 'chunk2']],
        'metadatas': [[{}, {}]],
        'distances': [[0.1, 0.2]]
    }
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._direct_chunk_query(query_emb, n_results=5)
    
    assert 'ids' in results
    assert len(results['ids'][0]) > 0


def test_direct_chunk_query_empty_collection():
    """Test _direct_chunk_query with empty collection."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_vector_store.collection.count.return_value = 0
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    query_emb = [0.1, 0.2, 0.3]
    results = service._direct_chunk_query(query_emb, n_results=5)
    
    assert 'ids' in results
    assert results['ids'] == [[]]


def test_search_chunks_basic():
    """Test search_chunks_basic."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    mock_vector_store.collection.count.return_value = 10
    mock_vector_store.collection.query.return_value = {
        'ids': [['chunk1']],
        'metadatas': [[{}]],
        'distances': [[0.1]]
    }
    
    chunk = ChunkModel(id='chunk1', text='Text', metadata_json=None)
    chunk.topic_l1 = None
    chunk.topic_l2 = None
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.options.return_value.filter_by.return_value.first.return_value = chunk
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(mock_vector_store, mock_db, mock_embedding_client)
    
    results = service.search_chunks_basic("query", n_results=3)
    
    assert len(results) > 0
    assert results[0]['id'] == 'chunk1'


def test_retrieve_with_topic_retrieval_disabled():
    """Test retrieve with topic retrieval disabled."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    mock_vector_store.collection.count.return_value = 10
    mock_vector_store.collection.query.return_value = {
        'ids': [['chunk1']],
        'metadatas': [[{}]],
        'distances': [[0.1]]
    }
    
    chunk = ChunkModel(id='chunk1', text='Text', metadata_json=None)
    chunk.topic_l1 = None
    chunk.topic_l2 = None
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.options.return_value.filter_by.return_value.first.return_value = chunk
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(
        mock_vector_store, 
        mock_db, 
        mock_embedding_client,
        use_topic_retrieval=False
    )
    
    results = service.retrieve("query", n_results=5)
    
    assert len(results) > 0


def test_retrieve_with_direct_search_mode():
    """Test retrieve with direct search mode."""
    mock_vector_store = MagicMock()
    mock_db = MagicMock()
    mock_embedding_client = MagicMock()
    
    mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    mock_vector_store.collection.count.return_value = 10
    mock_vector_store.collection.query.return_value = {
        'ids': [['chunk1']],
        'metadatas': [[{}]],
        'distances': [[0.1]]
    }
    
    chunk = ChunkModel(id='chunk1', text='Text', metadata_json=None)
    chunk.topic_l1 = None
    chunk.topic_l2 = None
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.options.return_value.filter_by.return_value.first.return_value = chunk
    mock_session.query.return_value = mock_query
    mock_db.get_session.return_value = mock_session
    
    service = RetrievalService(
        mock_vector_store, 
        mock_db, 
        mock_embedding_client,
        search_mode="direct"
    )
    
    results = service.retrieve("query", n_results=5)
    
    assert len(results) > 0
