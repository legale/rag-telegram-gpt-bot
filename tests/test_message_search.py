"""
Tests for message_search module.
"""

import pytest
from datetime import datetime
from src.core.message_search import (
    convert_search_results_to_dict,
    _convert_similarity_to_distance,
    _log_retrieval_distances,
    search_message_links,
    search_message_contents,
    _filter_by_threshold,
    _parse_msg_id,
    _format_message_parts,
    _prepare_message_parts
)
from src.core.domain import SearchResult, Chunk, Message
from src.core.hybrid_retrieval import HybridRetrievalService
from src.storage.db import Database, MessageModel


class TestConvertSearchResultsToDict:
    """Tests for convert_search_results_to_dict function."""
    
    def test_convert_empty_list(self):
        """Test converting empty list."""
        result = convert_search_results_to_dict([])
        assert result == []
    
    def test_convert_single_result(self):
        """Test converting single result."""
        chunk = Chunk(id="chunk1", text="Test", metadata={"chat_id": "chat1"})
        search_result = SearchResult(chunk=chunk, score=0.8, topics=None)
        results = convert_search_results_to_dict([search_result])
        
        assert len(results) == 1
        assert results[0]["id"] == "chunk1"
        assert abs(results[0]["distance"] - 0.2) < 0.01  # 1.0 - 0.8
        assert results[0]["metadata"]["chat_id"] == "chat1"
    
    def test_convert_multiple_results(self):
        """Test converting multiple results."""
        chunks = [
            Chunk(id=f"chunk{i}", text=f"Test {i}", metadata={})
            for i in range(3)
        ]
        search_results = [
            SearchResult(chunk=chunk, score=0.8, topics=None)
            for chunk in chunks
        ]
        results = convert_search_results_to_dict(search_results)
        
        assert len(results) == 3
        assert all(abs(r["distance"] - 0.2) < 0.01 for r in results)
    
    def test_convert_with_topics(self):
        """Test converting with topics."""
        chunk = Chunk(id="chunk1", text="Test", metadata={})
        topics = {"l1": "topic1", "l2": "topic2"}
        search_result = SearchResult(chunk=chunk, score=0.8, topics=topics)
        results = convert_search_results_to_dict([search_result])
        
        assert results[0]["metadata"]["topics"] == topics


class TestConvertSimilarityToDistance:
    """Tests for _convert_similarity_to_distance function."""
    
    def test_convert_score_one(self):
        """Test converting score 1.0."""
        result = _convert_similarity_to_distance(1.0)
        assert result == 0.0
    
    def test_convert_score_zero(self):
        """Test converting score 0.0."""
        result = _convert_similarity_to_distance(0.0)
        assert result == 1.0
    
    def test_convert_score_half(self):
        """Test converting score 0.5."""
        result = _convert_similarity_to_distance(0.5)
        assert result == 0.5


class TestLogRetrievalDistances:
    """Tests for _log_retrieval_distances function."""
    
    def test_log_empty_results(self):
        """Test logging empty results."""
        _log_retrieval_distances([], "query", "context")  # Should not raise
    
    def test_log_with_results(self):
        """Test logging with results."""
        results = [
            {"id": "chunk1", "distance": 0.5, "source": "vector"}
        ]
        _log_retrieval_distances(results, "query", "context")  # Should not raise


class TestSearchMessageLinks:
    """Tests for search_message_links function."""
    
    def test_search_empty_results(self, tmp_path):
        """Test search with empty results."""
        from src.app.bootstrap import create_hybrid_retrieval
        from src.core.embedding import LocalEmbeddingClient
        
        db = Database(f"sqlite:///{tmp_path}/test.db")
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        
        links = search_message_links(retrieval, db, "test query", top_k=3)
        assert links == []
    
    def test_search_with_results(self, tmp_path):
        """Test search with results."""
        from src.app.bootstrap import create_hybrid_retrieval
        from src.core.embedding import LocalEmbeddingClient
        
        db = Database(f"sqlite:///{tmp_path}/test.db")
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        
        # Even without data, should return empty list, not error
        links = search_message_links(retrieval, db, "test", top_k=3)
        assert isinstance(links, list)


class TestSearchMessageContents:
    """Tests for search_message_contents function."""
    
    def test_search_empty_results(self, tmp_path):
        """Test search with empty results."""
        from src.app.bootstrap import create_hybrid_retrieval
        from src.core.embedding import LocalEmbeddingClient
        
        db = Database(f"sqlite:///{tmp_path}/test.db")
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        
        results = search_message_contents(retrieval, db, "test query", top_k=3)
        assert results == []
    
    def test_search_with_threshold(self, tmp_path):
        """Test search with threshold."""
        from src.app.bootstrap import create_hybrid_retrieval
        from src.core.embedding import LocalEmbeddingClient
        
        db = Database(f"sqlite:///{tmp_path}/test.db")
        client = LocalEmbeddingClient(model="all-MiniLM-L6-v2")
        retrieval = create_hybrid_retrieval(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            embedding_client=client
        )
        
        results = search_message_contents(retrieval, db, "test", top_k=3, threshold=1.5)
        assert isinstance(results, list)


class TestFilterByThreshold:
    """Tests for _filter_by_threshold function."""
    
    def test_filter_empty(self):
        """Test filtering empty list."""
        result = _filter_by_threshold([], 1.5)
        assert result == []
    
    def test_filter_all_pass(self):
        """Test filtering where all pass."""
        results = [
            {"id": "chunk1", "distance": 0.5},
            {"id": "chunk2", "distance": 1.0}
        ]
        filtered = _filter_by_threshold(results, 1.5)
        assert len(filtered) == 2
    
    def test_filter_some_pass(self):
        """Test filtering where some pass."""
        results = [
            {"id": "chunk1", "distance": 0.5},
            {"id": "chunk2", "distance": 2.0}
        ]
        filtered = _filter_by_threshold(results, 1.5)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "chunk1"
    
    def test_filter_none_pass(self):
        """Test filtering where none pass."""
        results = [
            {"id": "chunk1", "distance": 2.0},
            {"id": "chunk2", "distance": 3.0}
        ]
        filtered = _filter_by_threshold(results, 1.5)
        assert len(filtered) == 0
    
    def test_filter_missing_distance(self):
        """Test filtering with missing distance."""
        results = [
            {"id": "chunk1"}  # no distance
        ]
        filtered = _filter_by_threshold(results, 1.5)
        assert len(filtered) == 0  # Missing distance treated as inf


class TestParseMsgId:
    """Tests for _parse_msg_id function."""
    
    def test_parse_composite_format(self):
        """Test parsing composite format."""
        result = _parse_msg_id("chat1_12345")
        assert result == 12345
    
    def test_parse_simple_format(self):
        """Test parsing simple format."""
        result = _parse_msg_id("12345")
        assert result == 12345
    
    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        result = _parse_msg_id("invalid")
        assert isinstance(result, int)  # Should return hash-based fallback


class TestFormatMessageParts:
    """Tests for _format_message_parts function."""
    
    def test_format_basic(self, tmp_path):
        """Test formatting basic message."""
        from src.storage.db import Database
        
        db = Database(f"sqlite:///{tmp_path}/test.db")
        session = db.get_session()
        try:
            msg = MessageModel(
                msg_id="chat1_12345",
                chat_id="chat1",
                from_id="user1",
                text="Test message",
                ts=datetime.now()
            )
            session.add(msg)
            session.commit()
            
            parts = _format_message_parts(msg, 12345, 0.5, "chunk1", 0, False)
            
            assert len(parts) > 0
            # Check that parts have required keys
            assert "content" in parts[0] or "text" in parts[0]
            assert parts[0]["distance"] == 0.5
        finally:
            session.close()


class TestPrepareMessageParts:
    """Tests for _prepare_message_parts function."""
    
    def test_prepare_empty_results(self, tmp_path):
        """Test preparing with empty results."""
        db = Database(f"sqlite:///{tmp_path}/test.db")
        result = _prepare_message_parts(db, [], False)
        assert result == []
    
    def test_prepare_with_results_no_messages(self, tmp_path):
        """Test preparing with results but no messages."""
        db = Database(f"sqlite:///{tmp_path}/test.db")
        results = [{"id": "nonexistent_chunk", "distance": 0.5}]
        result = _prepare_message_parts(db, results, False)
        assert result == []

