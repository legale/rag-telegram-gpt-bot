"""Tests for src/bot/command_parser.py"""

import pytest
from unittest.mock import Mock
from src.bot.command_parser import parse_find_command_args


class TestParseFindCommandArgs:
    """Tests for parse_find_command_args"""
    
    def test_parse_with_hybrid_method(self):
        """Test parsing with hybrid method"""
        rag_method, action, query = parse_find_command_args("hybrid test query")
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test query"
    
    def test_parse_with_vector_only_method(self):
        """Test parsing with vector_only method"""
        rag_method, action, query = parse_find_command_args("vector_only test query")
        
        assert rag_method == "vector_only"
        assert action is None
        assert query == "test query"
    
    def test_parse_with_fts_only_method(self):
        """Test parsing with fts_only method"""
        rag_method, action, query = parse_find_command_args("fts_only test query")
        
        assert rag_method == "fts_only"
        assert action is None
        assert query == "test query"
    
    def test_parse_with_list_action(self):
        """Test parsing with list action"""
        rag_method, action, query = parse_find_command_args("hybrid list")
        
        assert rag_method == "hybrid"
        assert action == "list"
        assert query is None
    
    def test_parse_with_slash_prefix(self):
        """Test parsing with /find prefix"""
        rag_method, action, query = parse_find_command_args("/find hybrid test query")
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test query"
    
    def test_parse_empty_text(self):
        """Test parsing empty text"""
        rag_method, action, query = parse_find_command_args("")
        
        assert rag_method is None
        assert action is None
        assert "Использование" in query
    
    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only text"""
        rag_method, action, query = parse_find_command_args("   ")
        
        assert rag_method is None
        assert action is None
        assert "Использование" in query
    
    def test_parse_invalid_method(self):
        """Test parsing with invalid method"""
        rag_method, action, query = parse_find_command_args("invalid_method test")
        
        assert rag_method is None
        assert action is None
        assert "Неизвестный метод RAG" in query
    
    def test_parse_missing_action_or_query(self):
        """Test parsing with missing action or query"""
        rag_method, action, query = parse_find_command_args("hybrid")
        
        assert rag_method is None
        assert action is None
        assert "Необходимо указать действие" in query
    
    def test_parse_single_word_query(self):
        """Test parsing single word query"""
        rag_method, action, query = parse_find_command_args("hybrid test")
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test"
    
    def test_parse_multi_word_query(self):
        """Test parsing multi-word query"""
        rag_method, action, query = parse_find_command_args("hybrid test query with multiple words")
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test query with multiple words"
    
    def test_parse_case_insensitive_method(self):
        """Test parsing is case-insensitive for method"""
        rag_method, action, query = parse_find_command_args("HYBRID test")
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test"
    
    def test_parse_case_insensitive_list(self):
        """Test parsing is case-insensitive for list action"""
        rag_method, action, query = parse_find_command_args("hybrid LIST")
        
        assert rag_method == "hybrid"
        assert action == "list"
        assert query is None
    
    def test_parse_empty_query_after_method(self):
        """Test parsing when query is empty after method"""
        rag_method, action, query = parse_find_command_args("hybrid ")
        
        assert rag_method is None
        assert action is None
        assert "Необходимо указать действие" in query or "Необходимо указать запрос" in query
    
    def test_parse_with_admin_manager_ignored(self):
        """Test that admin_manager parameter is accepted but not used"""
        mock_admin = Mock()
        rag_method, action, query = parse_find_command_args("hybrid test", mock_admin)
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test"
    
    def test_parse_with_default_threshold_ignored(self):
        """Test that default_threshold parameter is accepted but not used"""
        rag_method, action, query = parse_find_command_args("hybrid test", None, 2.0)
        
        assert rag_method == "hybrid"
        assert action is None
        assert query == "test"
