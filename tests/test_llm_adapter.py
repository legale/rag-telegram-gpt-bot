"""Tests for src/adapters/llm/llm_adapter.py"""

import pytest
from unittest.mock import Mock
from src.adapters.llm.llm_adapter import LLMAdapter


class TestLLMAdapter:
    """Tests for LLMAdapter"""
    
    def test_init(self):
        """Test LLMAdapter initialization"""
        mock_client = Mock()
        adapter = LLMAdapter(mock_client)
        
        assert adapter.llm_client == mock_client
    
    def test_complete_without_system(self):
        """Test complete without system message"""
        mock_client = Mock()
        mock_client.complete.return_value = "Response"
        adapter = LLMAdapter(mock_client)
        
        result = adapter.complete("test prompt")
        
        assert result == "Response"
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["role"] == "user"
        assert call_args[0]["content"] == "test prompt"
    
    def test_complete_with_system(self):
        """Test complete with system message"""
        mock_client = Mock()
        mock_client.complete.return_value = "Response"
        adapter = LLMAdapter(mock_client)
        
        result = adapter.complete("test prompt", system="System message")
        
        assert result == "Response"
        call_args = mock_client.complete.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
    
    def test_complete_with_kwargs(self):
        """Test complete with additional kwargs"""
        mock_client = Mock()
        mock_client.complete.return_value = "Response"
        adapter = LLMAdapter(mock_client)
        
        result = adapter.complete("test", temperature=0.9, max_tokens=2000)
        
        assert result == "Response"
        call_kwargs = mock_client.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["max_tokens"] == 2000
