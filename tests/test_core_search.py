"""
Tests for core/search module.
"""

import pytest
from datetime import datetime
from src.core.search import HybridSearch
from src.core.domain import Chunk, SearchResult
from src.core.interfaces import Embedder, VectorIndex, ChunkStore, MessageStore, ScoredDoc


class TestHybridSearch:
    """Tests for HybridSearch class."""
    
    def test_init(self):
        """Test initialization."""
        class MockEmbedder:
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockVectorIndex:
            def query(self, vector, top_k, filter=None):
                return []
        
        class MockChunkStore:
            def get_by_ids(self, ids):
                return []
        
        class MockMessageStore:
            def get_context(self, chat_id, time_point, window_sec):
                return []
        
        search = HybridSearch(
            embedder=MockEmbedder(),
            vector_index=MockVectorIndex(),
            chunk_store=MockChunkStore(),
            message_store=MockMessageStore()
        )
        assert search is not None
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        class MockEmbedder:
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockVectorIndex:
            def query(self, vector, top_k, filter=None):
                return []
        
        class MockChunkStore:
            def get_by_ids(self, ids):
                return []
        
        class MockMessageStore:
            def get_context(self, chat_id, time_point, window_sec):
                return []
        
        search = HybridSearch(
            embedder=MockEmbedder(),
            vector_index=MockVectorIndex(),
            chunk_store=MockChunkStore(),
            message_store=MockMessageStore()
        )
        
        result = search.search("")
        assert result == []
    
    def test_search_with_results(self):
        """Test search with results."""
        class MockEmbedder:
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockVectorIndex:
            def query(self, vector, top_k, filter=None):
                return [
                    ScoredDoc(id="chunk1", score=0.8, meta={})
                ]
        
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test chunk", metadata={})
                ]
        
        class MockMessageStore:
            def get_context(self, chat_id, time_point, window_sec):
                return []
        
        search = HybridSearch(
            embedder=MockEmbedder(),
            vector_index=MockVectorIndex(),
            chunk_store=MockChunkStore(),
            message_store=MockMessageStore()
        )
        
        results = search.search("test query")
        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"
        assert results[0].score == 0.8
    
    def test_search_with_threshold(self):
        """Test search with threshold filtering."""
        class MockEmbedder:
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockVectorIndex:
            def query(self, vector, top_k, filter=None):
                return [
                    ScoredDoc(id="chunk1", score=0.8, meta={}),
                    ScoredDoc(id="chunk2", score=0.3, meta={})
                ]
        
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test chunk 1", metadata={}),
                    Chunk(id="chunk2", text="Test chunk 2", metadata={})
                ]
        
        class MockMessageStore:
            def get_context(self, chat_id, time_point, window_sec):
                return []
        
        search = HybridSearch(
            embedder=MockEmbedder(),
            vector_index=MockVectorIndex(),
            chunk_store=MockChunkStore(),
            message_store=MockMessageStore()
        )
        
        results = search.search("test query", threshold=0.5)
        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"
    
    def test_search_with_chat_filter(self):
        """Test search with chat_id filter."""
        class MockEmbedder:
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockVectorIndex:
            def query(self, vector, top_k, filter=None):
                assert filter == {"chat_id": "chat1"}
                return [
                    ScoredDoc(id="chunk1", score=0.8, meta={})
                ]
        
        class MockChunkStore:
            def get_by_ids(self, ids):
                return [
                    Chunk(id="chunk1", text="Test chunk", metadata={"chat_id": "chat1"})
                ]
        
        class MockMessageStore:
            def get_context(self, chat_id, time_point, window_sec):
                return []
        
        search = HybridSearch(
            embedder=MockEmbedder(),
            vector_index=MockVectorIndex(),
            chunk_store=MockChunkStore(),
            message_store=MockMessageStore()
        )
        
        results = search.search("test query", chat_id="chat1")
        assert len(results) == 1

