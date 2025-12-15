"""
Tests for LLMAdapter.
"""

import pytest
from src.adapters.llm.llm_adapter import LLMAdapter
from src.core.llm import LLMClient


class TestLLMAdapter:
    """Tests for LLMAdapter class."""
    
    def test_init(self):
        """Test initialization."""
        # Create a simple test client without API key
        class TestLLMClient:
            def complete(self, messages, **kwargs):
                return "Test response"
        
        client = TestLLMClient()
        adapter = LLMAdapter(client)
        assert adapter.llm_client == client
    
    def test_complete_without_system(self):
        """Test complete method without system message."""
        class TestLLMClient:
            def complete(self, messages, **kwargs):
                return "Test response"
        
        client = TestLLMClient()
        adapter = LLMAdapter(client)
        result = adapter.complete("Hello")
        assert result == "Test response"
    
    def test_complete_with_system(self):
        """Test complete method with system message."""
        class TestLLMClient:
            def complete(self, messages, **kwargs):
                return "Test response"
        
        client = TestLLMClient()
        adapter = LLMAdapter(client)
        result = adapter.complete("Hello", system="You are a helpful assistant")
        assert result == "Test response"
    
    def test_complete_with_kwargs(self):
        """Test complete method with additional kwargs."""
        call_kwargs = {}
        class TestLLMClient:
            def complete(self, messages, **kwargs):
                call_kwargs.update(kwargs)
                return "Test response"
        
        client = TestLLMClient()
        adapter = LLMAdapter(client)
        result = adapter.complete("Hello", temperature=0.8, max_tokens=2000)
        assert result == "Test response"
        assert call_kwargs.get("temperature") == 0.8
        assert call_kwargs.get("max_tokens") == 2000

