import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import os

from src.core.topics import SimpleKMeans, TopicSummarizer
from src.storage.vector_store import VectorStore

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
