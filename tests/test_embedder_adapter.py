"""Tests for src/adapters/embedding/embedder_adapter.py"""

import pytest
from unittest.mock import Mock
from src.adapters.embedding.embedder_adapter import EmbedderAdapter


class TestEmbedderAdapter:
    """Tests for EmbedderAdapter"""
    
    def test_init(self):
        """Test EmbedderAdapter initialization"""
        mock_client = Mock()
        adapter = EmbedderAdapter(mock_client)
        
        assert adapter.client == mock_client
    
    def test_embed_documents(self):
        """Test embed_documents method"""
        mock_client = Mock()
        mock_client.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        adapter = EmbedderAdapter(mock_client)
        
        result = adapter.embed_documents(["text1", "text2"])
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.get_embeddings.assert_called_once_with(["text1", "text2"])
    
    def test_embed_query(self):
        """Test embed_query method"""
        mock_client = Mock()
        mock_client.get_embedding.return_value = [0.1, 0.2, 0.3]
        adapter = EmbedderAdapter(mock_client)
        
        result = adapter.embed_query("test query")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.get_embedding.assert_called_once_with("test query")
