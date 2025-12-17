"""Tests for src/core/message_search.py"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.core.message_search import (
    convert_search_results_to_dict,
    _convert_similarity_to_distance,
    _log_retrieval_distances,
    search_message_links,
    search_message_contents,
    _filter_results_by_threshold,
    _parse_msg_id,
    _format_message_parts,
    _prepare_message_parts_from_results
)
from src.core.domain import SearchResult, Chunk, Message
from src.storage.db import MessageModel
from src.lib.syslog2 import LOG_ALERT, LOG_DEBUG


class TestConvertSearchResultsToDict:
    """Tests for convert_search_results_to_dict"""
    
    def test_convert_with_metadata(self):
        """Test converting SearchResult with metadata"""
        chunk = Chunk(
            id="chunk1",
            text="test text",
            metadata={"key": "value"}
        )
        result = SearchResult(chunk=chunk, score=0.8)
        
        dict_results = convert_search_results_to_dict([result])
        
        assert len(dict_results) == 1
        assert dict_results[0]["id"] == "chunk1"
        assert abs(dict_results[0]["distance"] - 0.2) < 0.0001  # 1.0 - 0.8
        assert dict_results[0]["metadata"]["key"] == "value"
    
    def test_convert_with_topics(self):
        """Test converting SearchResult with topics"""
        chunk = Chunk(id="chunk1", text="test", metadata={})
        result = SearchResult(chunk=chunk, score=0.9, topics=["topic1", "topic2"])
        
        dict_results = convert_search_results_to_dict([result])
        
        assert dict_results[0]["metadata"]["topics"] == ["topic1", "topic2"]
    
    def test_convert_multiple_results(self):
        """Test converting multiple SearchResults"""
        results = [
            SearchResult(chunk=Chunk(id="chunk1", text="test1"), score=0.8),
            SearchResult(chunk=Chunk(id="chunk2", text="test2"), score=0.7),
        ]
        
        dict_results = convert_search_results_to_dict(results)
        
        assert len(dict_results) == 2
        assert dict_results[0]["id"] == "chunk1"
        assert dict_results[1]["id"] == "chunk2"
    
    def test_convert_score_above_one(self):
        """Test converting SearchResult with score > 1.0"""
        chunk = Chunk(id="chunk1", text="test")
        result = SearchResult(chunk=chunk, score=1.5)
        
        dict_results = convert_search_results_to_dict([result])
        
        assert dict_results[0]["distance"] == 0.0  # 1.0 - 1.5 = -0.5, but clamped to 0.0


class TestConvertSimilarityToDistance:
    """Tests for _convert_similarity_to_distance"""
    
    def test_convert_normal_score(self):
        """Test converting normal similarity score"""
        distance = _convert_similarity_to_distance(0.8)
        assert abs(distance - 0.2) < 0.0001
    
    def test_convert_zero_score(self):
        """Test converting zero similarity score"""
        distance = _convert_similarity_to_distance(0.0)
        assert distance == 1.0
    
    def test_convert_one_score(self):
        """Test converting score of 1.0"""
        distance = _convert_similarity_to_distance(1.0)
        assert distance == 0.0


class TestLogRetrievalDistances:
    """Tests for _log_retrieval_distances"""
    
    def test_log_empty_results(self):
        """Test logging with empty results"""
        with patch('src.core.message_search.syslog2') as mock_syslog:
            _log_retrieval_distances([], "query", "context")
            mock_syslog.assert_not_called()
    
    def test_log_with_results(self):
        """Test logging with results"""
        results = [
            {"id": "chunk1", "distance": 0.5},
            {"id": "chunk2", "distance": 0.3}
        ]
        
        with patch('src.core.message_search.syslog2') as mock_syslog:
            _log_retrieval_distances(results, "test query", "contents")
            
            # Should log summary and each result
            assert mock_syslog.call_count >= 3
    
    def test_log_with_debug_rag(self):
        """Test logging with debug_rag enabled"""
        results = [{"id": "chunk1", "distance": 0.5, "metadata": {"key": "value"}}]
        
        with patch('src.core.message_search.syslog2') as mock_syslog:
            _log_retrieval_distances(results, "query", "context", debug_rag=True)
            
            # Should have more log calls with debug enabled
            assert mock_syslog.call_count >= 3


class TestSearchMessageLinks:
    """Tests for search_message_links"""
    
    def test_search_no_results(self):
        """Test search with no results"""
        mock_retrieval = Mock()
        mock_retrieval.search_chunks_basic.return_value = []
        mock_db = Mock()
        
        links = search_message_links(mock_retrieval, mock_db, "query")
        
        assert links == []
    
    def test_search_with_results(self):
        """Test search with results"""
        mock_retrieval = Mock()
        mock_retrieval.search_chunks_basic.return_value = [
            {"id": "chunk1"},
            {"id": "chunk2"}
        ]
        mock_db = Mock()
        mock_db.get_chunk_link_info.side_effect = [
            ("chat1", "msg1", "username1"),
            ("chat2", "msg2", "username2")
        ]
        
        with patch('src.core.message_search.build_message_link') as mock_build:
            mock_build.side_effect = ["link1", "link2"]
            links = search_message_links(mock_retrieval, mock_db, "query", top_k=2)
            
            assert len(links) == 2
            assert links == ["link1", "link2"]
    
    def test_search_skips_chunks_without_id(self):
        """Test search skips chunks without id"""
        mock_retrieval = Mock()
        mock_retrieval.search_chunks_basic.return_value = [
            {"id": None},
            {"id": "chunk1"}
        ]
        mock_db = Mock()
        mock_db.get_chunk_link_info.return_value = ("chat1", "msg1", "username1")
        
        with patch('src.core.message_search.build_message_link') as mock_build:
            mock_build.return_value = "link1"
            links = search_message_links(mock_retrieval, mock_db, "query")
            
            assert len(links) == 1


class TestSearchMessageContents:
    """Tests for search_message_contents"""
    
    def test_search_without_threshold(self):
        """Test search without threshold"""
        mock_retrieval = Mock()
        mock_retrieval.search_chunks_basic.return_value = [{"id": "chunk1", "distance": 0.5}]
        mock_db = Mock()
        mock_db.get_messages_by_chunk.return_value = []
        
        with patch('src.core.message_search._prepare_message_parts_from_results') as mock_prepare:
            mock_prepare.return_value = []
            result = search_message_contents(mock_retrieval, mock_db, "query")
            
            assert result == []
            mock_retrieval.search_chunks_basic.assert_called_once_with("query", n_results=3)
    
    def test_search_with_threshold(self):
        """Test search with threshold"""
        mock_retrieval = Mock()
        mock_retrieval.search_chunks_basic.return_value = [
            {"id": "chunk1", "distance": 0.3},
            {"id": "chunk2", "distance": 0.8}
        ]
        mock_db = Mock()
        
        with patch('src.core.message_search._filter_results_by_threshold') as mock_filter, \
             patch('src.core.message_search._prepare_message_parts_from_results') as mock_prepare:
            mock_filter.return_value = [{"id": "chunk1", "distance": 0.3}]
            mock_prepare.return_value = []
            
            result = search_message_contents(
                mock_retrieval, mock_db, "query", top_k=10, threshold=0.5
            )
            
            mock_filter.assert_called_once()
            assert mock_filter.call_args[0][1] == 0.5  # threshold


class TestFilterResultsByThreshold:
    """Tests for _filter_results_by_threshold"""
    
    def test_filter_all_pass(self):
        """Test filtering when all results pass threshold"""
        results = [
            {"id": "chunk1", "distance": 0.3},
            {"id": "chunk2", "distance": 0.5}
        ]
        
        filtered = _filter_results_by_threshold(results, 1.0)
        
        assert len(filtered) == 2
    
    def test_filter_some_filtered(self):
        """Test filtering when some results are filtered"""
        results = [
            {"id": "chunk1", "distance": 0.3},
            {"id": "chunk2", "distance": 0.8},
            {"id": "chunk3", "distance": 1.2}
        ]
        
        filtered = _filter_results_by_threshold(results, 0.5)
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "chunk1"
    
    def test_filter_with_debug_rag(self):
        """Test filtering with debug_rag enabled"""
        results = [
            {"id": "chunk1", "distance": 0.3},
            {"id": "chunk2", "distance": 0.8}
        ]
        
        with patch('src.core.message_search.syslog2') as mock_syslog:
            filtered = _filter_results_by_threshold(results, 0.5, debug_rag=True)
            
            assert len(filtered) == 1
            mock_syslog.assert_called()
    
    def test_filter_missing_distance(self):
        """Test filtering when distance is missing"""
        results = [
            {"id": "chunk1"},  # no distance
            {"id": "chunk2", "distance": 0.3}
        ]
        
        filtered = _filter_results_by_threshold(results, 0.5)
        
        # Missing distance should be treated as inf, so filtered out
        assert len(filtered) == 1
        assert filtered[0]["id"] == "chunk2"


class TestParseMsgId:
    """Tests for _parse_msg_id"""
    
    def test_parse_composite_format(self):
        """Test parsing composite format msg_id"""
        msg_id = _parse_msg_id("chat123_456")
        assert msg_id == 456
    
    def test_parse_simple_format(self):
        """Test parsing simple numeric msg_id"""
        msg_id = _parse_msg_id("123")
        assert msg_id == 123
    
    def test_parse_invalid_format(self):
        """Test parsing invalid format falls back to hash"""
        msg_id = _parse_msg_id("invalid")
        # Should return a hash-based number
        assert isinstance(msg_id, int)
        assert 0 <= msg_id < 10 ** 9


class TestFormatMessageParts:
    """Tests for _format_message_parts"""
    
    def test_format_basic_message(self):
        """Test formatting basic message"""
        msg = MessageModel(
            msg_id="chat123_456",
            chat_id="chat123",
            from_id="user1",
            text="Test message",
            ts=datetime.now()
        )
        
        with patch('src.core.message_search.split_message_if_needed') as mock_split:
            mock_split.return_value = [{"text": "Test message"}]
            parts = _format_message_parts(msg, 456, 0.5, "chunk1", 0, False)
            
            assert len(parts) == 1
            assert parts[0]["distance"] == 0.5
            mock_split.assert_called_once()
    
    def test_format_with_debug_rag(self):
        """Test formatting with debug_rag enabled"""
        msg = MessageModel(
            msg_id="chat123_456",
            chat_id="chat123",
            from_id="user1",
            text="Test",
            ts=datetime.now()
        )
        
        with patch('src.core.message_search.split_message_if_needed') as mock_split, \
             patch('src.core.message_search.syslog2') as mock_syslog:
            mock_split.return_value = [{"text": "Test"}]
            parts = _format_message_parts(msg, 456, 0.5, "chunk1", 0, True)
            
            assert len(parts) == 1
            mock_syslog.assert_called()
    
    def test_format_propagates_distance(self):
        """Test that distance is propagated to all parts"""
        msg = MessageModel(
            msg_id="chat123_456",
            chat_id="chat123",
            from_id="user1",
            text="Test",
            ts=datetime.now()
        )
        
        with patch('src.core.message_search.split_message_if_needed') as mock_split:
            mock_split.return_value = [
                {"text": "Part 1"},
                {"text": "Part 2"}
            ]
            parts = _format_message_parts(msg, 456, 0.7, "chunk1", 0, False)
            
            assert len(parts) == 2
            assert parts[0]["distance"] == 0.7
            assert parts[1]["distance"] == 0.7


class TestPrepareMessagePartsFromResults:
    """Tests for _prepare_message_parts_from_results"""
    
    def test_prepare_no_results(self):
        """Test preparing with no results"""
        mock_db = Mock()
        result = _prepare_message_parts_from_results(mock_db, [], False)
        
        assert result == []
    
    def test_prepare_with_results(self):
        """Test preparing with results"""
        mock_db = Mock()
        msg = MessageModel(
            msg_id="chat123_456",
            chat_id="chat123",
            from_id="user1",
            text="Test message",
            ts=datetime.now()
        )
        mock_db.get_messages_by_chunk.return_value = [msg]
        
        results = [{"id": "chunk1", "distance": 0.5}]
        
        with patch('src.core.message_search._parse_msg_id') as mock_parse, \
             patch('src.core.message_search._format_message_parts') as mock_format:
            mock_parse.return_value = 456
            mock_format.return_value = [{"text": "Test message", "distance": 0.5}]
            
            parts = _prepare_message_parts_from_results(mock_db, results, False)
            
            assert len(parts) == 1
            assert len(parts[0]) == 1
    
    def test_prepare_skips_chunks_without_id(self):
        """Test preparing skips chunks without id"""
        mock_db = Mock()
        results = [
            {"id": None, "distance": 0.5},
            {"id": "chunk1", "distance": 0.3}
        ]
        
        with patch('src.core.message_search._prepare_message_parts_from_results') as mock_prepare:
            # This is recursive, so we'll test the skip logic directly
            msg = MessageModel(
                msg_id="chat123_456",
                chat_id="chat123",
                from_id="user1",
                text="Test",
                ts=datetime.now()
            )
            mock_db.get_messages_by_chunk.return_value = [msg]
            
            with patch('src.core.message_search._parse_msg_id') as mock_parse, \
                 patch('src.core.message_search._format_message_parts') as mock_format:
                mock_parse.return_value = 456
                mock_format.return_value = [{"text": "Test"}]
                
                parts = _prepare_message_parts_from_results(mock_db, results, False)
                
                # Should only process chunk1, skip None
                assert len(parts) == 1
    
    def test_prepare_with_debug_rag(self):
        """Test preparing with debug_rag enabled"""
        mock_db = Mock()
        msg = MessageModel(
            msg_id="chat123_456",
            chat_id="chat123",
            from_id="user1",
            text="Test",
            ts=datetime.now()
        )
        mock_db.get_messages_by_chunk.return_value = [msg]
        
        results = [{"id": "chunk1", "distance": 0.5}]
        
        with patch('src.core.message_search._parse_msg_id') as mock_parse, \
             patch('src.core.message_search._format_message_parts') as mock_format, \
             patch('src.core.message_search.syslog2') as mock_syslog:
            mock_parse.return_value = 456
            mock_format.return_value = [{"text": "Test"}]
            
            parts = _prepare_message_parts_from_results(mock_db, results, True)
            
            assert len(parts) == 1
            mock_syslog.assert_called()
    
    def test_prepare_skips_empty_messages(self):
        """Test preparing skips chunks with no messages"""
        mock_db = Mock()
        mock_db.get_messages_by_chunk.return_value = []
        
        results = [{"id": "chunk1", "distance": 0.5}]
        
        parts = _prepare_message_parts_from_results(mock_db, results, False)
        
        assert parts == []
