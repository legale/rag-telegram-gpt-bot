"""Tests for src/core/search.py"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
from src.core.search import HybridSearch
from src.core.domain import Chunk, Message, SearchResult


class TestHybridSearchInit:
    """Tests for HybridSearch.__init__"""
    
    def test_init(self):
        """Test HybridSearch initialization"""
        mock_embedder = Mock()
        mock_vector_index = Mock()
        mock_chunk_store = Mock()
        mock_message_store = Mock()
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=mock_message_store
        )
        
        assert search.embedder == mock_embedder
        assert search.vector_index == mock_vector_index
        assert search.chunk_store == mock_chunk_store
        assert search.message_store == mock_message_store


class TestHybridSearchSearch:
    """Tests for HybridSearch.search"""
    
    def test_search_empty_query(self):
        """Test search with empty query"""
        search = HybridSearch(
            embedder=Mock(),
            vector_index=Mock(),
            chunk_store=Mock(),
            message_store=Mock()
        )
        
        results = search.search("")
        
        assert results == []
    
    def test_search_whitespace_query(self):
        """Test search with whitespace-only query"""
        search = HybridSearch(
            embedder=Mock(),
            vector_index=Mock(),
            chunk_store=Mock(),
            message_store=Mock()
        )
        
        results = search.search("   ")
        
        assert results == []
    
    def test_search_no_results(self):
        """Test search with no results"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = []
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=Mock(),
            message_store=Mock()
        )
        
        results = search.search("test query")
        
        assert results == []
    
    def test_search_with_results(self):
        """Test search with results"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9),
            Mock(id="chunk2", score=0.8)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(id="chunk1", text="Text 1"),
            Chunk(id="chunk2", text="Text 2")
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query")
        
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.8
        # Results should be sorted by score descending
        assert results[0].score >= results[1].score
    
    def test_search_with_threshold(self):
        """Test search with threshold filtering"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9),
            Mock(id="chunk2", score=0.5),
            Mock(id="chunk3", score=0.3)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(id="chunk1", text="Text 1"),
            Chunk(id="chunk2", text="Text 2"),
            Chunk(id="chunk3", text="Text 3")
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query", threshold=0.6)
        
        assert len(results) == 1
        assert results[0].score == 0.9
    
    def test_search_with_chat_id_filter(self):
        """Test search with chat_id filter"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(id="chunk1", text="Text 1")
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query", chat_id="chat1")
        
        # Verify filter was passed to vector_index.query
        call_args = mock_vector_index.query.call_args
        assert call_args[1]["filter"] == {"chat_id": "chat1"}
    
    def test_search_enriches_with_messages(self):
        """Test search enriches results with messages"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9)
        ]
        time_point = datetime.now()
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(
                id="chunk1",
                text="Text 1",
                valid_period=(time_point, time_point),
                metadata={"chat_id": "chat1"}
            )
        ]
        mock_message_store = Mock()
        mock_message_store.get_context.return_value = [
            Message(
                id="msg1",
                chat_id="chat1",
                from_id="user1",
                text="Message",
                timestamp=time_point
            )
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=mock_message_store
        )
        
        results = search.search("test query", enrich_with_messages=True)
        
        assert len(results) == 1
        assert len(results[0].original_messages) == 1
    
    def test_search_without_message_enrichment(self):
        """Test search without message enrichment"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(id="chunk1", text="Text 1")
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query", enrich_with_messages=False)
        
        assert len(results) == 1
        assert len(results[0].original_messages) == 0
    
    def test_search_skips_missing_chunks(self):
        """Test search skips chunks not found in store"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9),
            Mock(id="chunk2", score=0.8)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(id="chunk1", text="Text 1")
            # chunk2 is missing
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query")
        
        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"
    
    def test_search_extracts_topics_from_metadata(self):
        """Test search extracts topics from chunk metadata"""
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = [0.1] * 10
        mock_vector_index = Mock()
        mock_vector_index.query.return_value = [
            Mock(id="chunk1", score=0.9)
        ]
        mock_chunk_store = Mock()
        mock_chunk_store.get_by_ids.return_value = [
            Chunk(
                id="chunk1",
                text="Text 1",
                metadata={"topic_ids": ["topic1", "topic2"]}
            )
        ]
        
        search = HybridSearch(
            embedder=mock_embedder,
            vector_index=mock_vector_index,
            chunk_store=mock_chunk_store,
            message_store=Mock()
        )
        
        results = search.search("test query")
        
        assert len(results) == 1
        assert "topic1" in results[0].topics
        assert "topic2" in results[0].topics

