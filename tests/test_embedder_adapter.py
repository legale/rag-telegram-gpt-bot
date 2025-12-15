"""
Tests for EmbedderAdapter.
"""

import pytest
from src.adapters.embedding.embedder_adapter import EmbedderAdapter
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient


class TestEmbedderAdapter:
    """Tests for EmbedderAdapter class."""
    
    def test_init_with_embedding_client(self):
        """Test initialization with EmbeddingClient."""
        # Use LocalEmbeddingClient to avoid API key requirement
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        adapter = EmbedderAdapter(client)
        assert adapter.client == client
    
    def test_init_with_local_embedding_client(self):
        """Test initialization with LocalEmbeddingClient."""
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        adapter = EmbedderAdapter(client)
        assert adapter.client == client
    
    def test_embed_documents(self):
        """Test embed_documents method."""
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        adapter = EmbedderAdapter(client)
        
        texts = ["Hello world", "Test text"]
        embeddings = adapter.embed_documents(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert len(embeddings[1]) > 0
    
    def test_embed_query(self):
        """Test embed_query method."""
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        adapter = EmbedderAdapter(client)
        
        query = "test query"
        embedding = adapter.embed_query(query)
        
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

