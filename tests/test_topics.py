import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import os

from src.core.topics import SimpleKMeans, TopicSummarizer, TopicBuilder
from src.storage.db import Database, TopicModel, TopicChunkModel
from src.storage.vector_store import VectorStore

# --- DB Tests ---

@pytest.fixture
def mock_db_session():
    with patch('src.storage.db.Database.get_session') as mock_get:
        session = MagicMock()
        mock_get.return_value = session
        yield session

def test_db_create_topic(mock_db_session):
    db = Database("sqlite:///:memory:")
    
    # Mock behavior
    mock_db_session.add.side_effect = lambda x: setattr(x, 'id', 1)
    
    t_id = db.create_topic("Title", "Description")
    
    assert t_id == 1
    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()

def test_db_add_topic_chunks(mock_db_session):
    db = Database("sqlite:///:memory:")
    
    count = db.add_topic_chunks(1, ["c1", "c2"])
    
    assert count == 2
    mock_db_session.add_all.assert_called()
    mock_db_session.commit.assert_called()

# --- SimpleKMeans Tests ---

def test_simple_kmeans_fit():
    # Create simple 2D data: 3 clusters
    X = np.vstack([
        np.random.normal(0, 0.1, (10, 2)),
        np.random.normal(5, 0.1, (10, 2)),
        np.random.normal(10, 0.1, (10, 2))
    ])
    
    kmeans = SimpleKMeans(n_clusters=3, seed=42)
    kmeans.fit(X)
    
    assert kmeans.centroids is not None
    assert len(kmeans.centroids) == 3
    assert len(kmeans.labels_) == 30
    # Check that labels are somewhat consistent (0-2)
    assert set(kmeans.labels_) == {0, 1, 2}

def test_simple_kmeans_small_data():
    # Fewer points than clusters
    X = np.random.rand(2, 5)
    kmeans = SimpleKMeans(n_clusters=5)
    kmeans.fit(X)
    
    assert len(kmeans.centroids) == 2
    assert len(kmeans.labels_) == 2

# --- TopicSummarizer Tests ---

def test_topic_summarizer():
    mock_llm = MagicMock()
    summarizer = TopicSummarizer(mock_llm)
    
    # Mock response
    mock_llm.complete.return_value = '{"title": "Test Topic", "description": "A test description."}'
    
    title, desc = summarizer.generate_topic_title_and_description(["chunk1", "chunk2"])
    
    assert title == "Test Topic"
    assert desc == "A test description."
    mock_llm.complete.assert_called_once()

def test_topic_summarizer_error_handling():
    mock_llm = MagicMock()
    summarizer = TopicSummarizer(mock_llm)
    mock_llm.complete.side_effect = Exception("LLM Error")
    
    title, desc = summarizer.generate_topic_title_and_description(["chunk1"])
    
    assert title == "Unknown Topic"

# --- TopicBuilder Tests ---

@patch("src.core.topics.Database")
@patch("src.core.topics.VectorStore")
@patch("src.core.topics.LLMClient")
def test_topic_builder_pipeline(MockLLM, MockVectorStore, MockDB):
    builder = TopicBuilder("sqlite:///", "chroma/")
    
    # Mock Vector Store
    mock_store = MockVectorStore.return_value
    # 10 vectors, 2 dims
    mock_store.get_all_embeddings.return_value = {
        "embeddings": np.random.rand(10, 2).tolist(),
        "ids": [f"c{i}" for i in range(10)]
    }
    
    # Mock DB
    mock_db = MockDB.return_value
    mock_db.get_chunk_text.return_value = "Sample text content"
    mock_db.create_topic.return_value = 1
    
    # Mock LLM
    mock_llm_instance = MockLLM.return_value
    mock_llm_instance.complete.return_value = '{"title": "T", "description": "D"}'
    
    # Run
    count = builder.build_topics(clear_existing=True)
    
    # Assertions
    assert count > 0 # Should find at least 1 cluster (k approx sqrt(5) = 2)
    mock_db.clear_topics.assert_called_once()
    mock_db.create_topic.assert_called()
    mock_db.add_topic_chunks.assert_called()

