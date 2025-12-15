"""
Tests for query_rewriter module.
"""

import pytest
from src.core.query_rewriter import QueryRewriter
from src.core.llm import LLMClient
from src.lib.syslog2 import LOG_WARNING


class TestQueryRewriter:
    """Tests for QueryRewriter class."""
    
    def test_init_without_llm(self):
        """Test initialization without LLM."""
        rewriter = QueryRewriter()
        assert rewriter.llm is None
        assert rewriter.log_level == LOG_WARNING
    
    def test_init_with_llm(self):
        """Test initialization with LLM."""
        # Create a simple test LLM without API key
        class TestLLM:
            def complete(self, prompt, system=None, **kwargs):
                return "test response"
        
        llm = TestLLM()
        rewriter = QueryRewriter(llm=llm)
        assert rewriter.llm == llm
    
    def test_expand_without_llm(self):
        """Test expand without LLM."""
        rewriter = QueryRewriter()
        result = rewriter.expand("test query")
        assert result == ["test query"]
    
    def test_rephrase_for_embedding_without_llm(self):
        """Test rephrase_for_embedding without LLM."""
        rewriter = QueryRewriter()
        result = rewriter.rephrase_for_embedding("test query")
        assert result == "test query"
    
    def test_expand_with_llm_error(self):
        """Test expand with LLM that raises error."""
        class FailingLLM:
            def complete(self, prompt, system=None, **kwargs):
                raise Exception("LLM error")
        
        rewriter = QueryRewriter(llm=FailingLLM())
        result = rewriter.expand("test query")
        assert result == ["test query"]  # Should fallback to original
    
    def test_rephrase_for_embedding_with_llm_error(self):
        """Test rephrase_for_embedding with LLM that raises error."""
        class FailingLLM:
            def complete(self, prompt, system=None, **kwargs):
                raise Exception("LLM error")
        
        rewriter = QueryRewriter(llm=FailingLLM())
        result = rewriter.rephrase_for_embedding("test query")
        assert result == "test query"  # Should fallback to original

