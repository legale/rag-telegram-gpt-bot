"""
Tests for core/prompt module.
"""

import pytest
from src.core.prompt import PromptEngine


class TestPromptEngine:
    """Tests for PromptEngine class."""
    
    def test_init(self):
        """Test initialization."""
        engine = PromptEngine()
        assert engine is not None
    
    def test_construct_prompt_empty_context(self):
        """Test construct_prompt with empty context."""
        engine = PromptEngine()
        result = engine.construct_prompt(
            context_chunks=[],
            chat_history=[],
            user_task="test task"
        )
        
        assert "test task" in result
        assert "Нет релевантного контекста" in result or "Нет недавних сообщений" in result
    
    def test_construct_prompt_with_context(self):
        """Test construct_prompt with context chunks."""
        engine = PromptEngine()
        context_chunks = [
            {"id": "chunk1", "text": "Test chunk text", "metadata": {}}
        ]
        result = engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=[],
            user_task="test task"
        )
        
        assert "test task" in result
        assert "Test chunk text" in result
    
    def test_construct_prompt_with_history(self):
        """Test construct_prompt with chat history."""
        engine = PromptEngine()
        chat_history = [
            {"sender": "User", "content": "Hello"}
        ]
        result = engine.construct_prompt(
            context_chunks=[],
            chat_history=chat_history,
            user_task="test task"
        )
        
        assert "test task" in result
        assert "User" in result
        assert "Hello" in result
    
    def test_construct_prompt_with_max_chars(self):
        """Test construct_prompt with max_context_chars limit."""
        engine = PromptEngine()
        # Create a large chunk
        large_text = "x" * 10000
        context_chunks = [
            {"id": "chunk1", "text": large_text, "metadata": {}}
        ]
        result = engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=[],
            user_task="test task",
            max_context_chars=1000
        )
        
        assert len(result) < 2000  # Should be truncated
    
    def test_construct_prompt_with_custom_template(self):
        """Test construct_prompt with custom template."""
        engine = PromptEngine()
        custom_template = "Custom: {task}\nContext: {context}\nHistory: {history}"
        result = engine.construct_prompt(
            context_chunks=[],
            chat_history=[],
            user_task="test task",
            custom_template=custom_template
        )
        
        assert "Custom:" in result
        assert "test task" in result
    
    def test_construct_prompt_chunk_without_text(self):
        """Test construct_prompt with chunk without text."""
        engine = PromptEngine()
        context_chunks = [
            {"id": "chunk1", "metadata": {}}  # No text field
        ]
        result = engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=[],
            user_task="test task"
        )
        
        # Should handle missing text gracefully
        assert "test task" in result
    
    def test_construct_prompt_with_metadata_string(self):
        """Test construct_prompt with string metadata."""
        import json
        engine = PromptEngine()
        metadata_str = json.dumps({"key": "value"})
        context_chunks = [
            {"id": "chunk1", "text": "Test", "metadata": metadata_str}
        ]
        result = engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=[],
            user_task="test task"
        )
        
        assert "Test" in result

