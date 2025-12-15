"""Tests for src/core/query_rewriter.py"""

import pytest
from unittest.mock import Mock, patch
from src.core.query_rewriter import QueryRewriter
from src.lib.syslog2 import LOG_WARNING, LOG_INFO


class TestQueryRewriterInit:
    """Tests for QueryRewriter.__init__"""
    
    def test_init_with_llm(self):
        """Test initialization with LLM"""
        mock_llm = Mock()
        rewriter = QueryRewriter(llm=mock_llm)
        
        assert rewriter.llm == mock_llm
        assert rewriter.log_level == LOG_WARNING
    
    def test_init_without_llm(self):
        """Test initialization without LLM"""
        rewriter = QueryRewriter()
        
        assert rewriter.llm is None
        assert rewriter.log_level == LOG_WARNING
    
    def test_init_with_log_level(self):
        """Test initialization with custom log level"""
        mock_llm = Mock()
        rewriter = QueryRewriter(llm=mock_llm, log_level=LOG_INFO)
        
        assert rewriter.log_level == LOG_INFO


class TestQueryRewriterExpand:
    """Tests for QueryRewriter.expand"""
    
    def test_expand_without_llm(self):
        """Test expand without LLM returns original only"""
        rewriter = QueryRewriter()
        
        variants = rewriter.expand("test query")
        
        assert variants == ["test query"]
    
    def test_expand_with_llm_success(self):
        """Test expand with LLM success"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "variant1\nvariant2\nvariant3"
        rewriter = QueryRewriter(llm=mock_llm)
        
        variants = rewriter.expand("original")
        
        assert "original" in variants
        assert len(variants) <= 4  # Max 4 variants
        mock_llm.complete.assert_called_once()
    
    def test_expand_with_llm_removes_duplicates(self):
        """Test expand removes duplicate variants"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "original\nvariant1\noriginal"
        rewriter = QueryRewriter(llm=mock_llm)
        
        variants = rewriter.expand("original")
        
        assert variants.count("original") == 1
    
    def test_expand_with_llm_case_insensitive_duplicates(self):
        """Test expand removes case-insensitive duplicates"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "Variant1\nvariant1"
        rewriter = QueryRewriter(llm=mock_llm)
        
        variants = rewriter.expand("original")
        
        # Should only have one variant1 (case-insensitive)
        variant_lower = [v.lower() for v in variants]
        assert variant_lower.count("variant1") == 1
    
    def test_expand_with_llm_exception(self):
        """Test expand handles LLM exception"""
        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")
        rewriter = QueryRewriter(llm=mock_llm)
        
        with patch('src.core.query_rewriter.syslog2') as mock_syslog:
            variants = rewriter.expand("test")
            
            assert variants == ["test"]
            mock_syslog.assert_called()
    
    def test_expand_with_llm_empty_response(self):
        """Test expand with empty LLM response"""
        mock_llm = Mock()
        mock_llm.complete.return_value = ""
        rewriter = QueryRewriter(llm=mock_llm)
        
        variants = rewriter.expand("original")
        
        assert variants == ["original"]


class TestQueryRewriterRephraseForEmbedding:
    """Tests for QueryRewriter.rephrase_for_embedding"""
    
    def test_rephrase_without_llm(self):
        """Test rephrase without LLM returns original"""
        rewriter = QueryRewriter()
        
        result = rewriter.rephrase_for_embedding("test query")
        
        assert result == "test query"
    
    def test_rephrase_with_llm_success(self):
        """Test rephrase with LLM success"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "Rephrased query"
        rewriter = QueryRewriter(llm=mock_llm)
        
        result = rewriter.rephrase_for_embedding("original query")
        
        assert result == "Rephrased query"
        mock_llm.complete.assert_called_once()
    
    def test_rephrase_with_llm_empty_response(self):
        """Test rephrase with empty LLM response returns original"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "   "
        rewriter = QueryRewriter(llm=mock_llm)
        
        result = rewriter.rephrase_for_embedding("original")
        
        assert result == "original"
    
    def test_rephrase_with_llm_exception(self):
        """Test rephrase handles LLM exception"""
        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")
        rewriter = QueryRewriter(llm=mock_llm)
        
        with patch('src.core.query_rewriter.syslog2') as mock_syslog:
            result = rewriter.rephrase_for_embedding("test")
            
            assert result == "test"
            mock_syslog.assert_called()
    
    def test_rephrase_strips_whitespace(self):
        """Test rephrase strips whitespace from response"""
        mock_llm = Mock()
        mock_llm.complete.return_value = "  Rephrased query  "
        rewriter = QueryRewriter(llm=mock_llm)
        
        result = rewriter.rephrase_for_embedding("original")
        
        assert result == "Rephrased query"
