"""
Tests for LegaleBot chat flow functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.bot.core import LegaleBot


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for LegaleBot."""
    with patch('src.bot.core.Database') as mock_db, \
         patch('src.bot.core.VectorStore') as mock_vs, \
         patch('src.bot.core.EmbeddingClient') as mock_ec, \
         patch('src.bot.core.LLMClient') as mock_llm, \
         patch('src.bot.core.RetrievalService') as mock_rs, \
         patch('src.bot.core.PromptEngine') as mock_pe:
        
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm_instance.count_tokens.return_value = 100
        mock_llm_instance.complete.return_value = "Test response"
        mock_llm.return_value = mock_llm_instance
        
        mock_rs_instance = MagicMock()
        mock_rs_instance.retrieve.return_value = ["Context chunk 1", "Context chunk 2"]
        mock_rs.return_value = mock_rs_instance
        
        mock_pe_instance = MagicMock()
        mock_pe_instance.construct_prompt.return_value = "System prompt with context"
        mock_pe.return_value = mock_pe_instance
        
        yield {
            'db': mock_db,
            'vector_store': mock_vs,
            'embedding_client': mock_ec,
            'llm_client': mock_llm,
            'llm_instance': mock_llm_instance,
            'retrieval_service': mock_rs,
            'retrieval_instance': mock_rs_instance,
            'prompt_engine': mock_pe,
            'prompt_instance': mock_pe_instance
        }


class TestChatBasic:
    """Tests for basic chat() functionality."""
    
    def test_chat_basic_query(self, mock_dependencies):
        """Test basic chat query with mocked retrieval."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        response = bot.chat("What is the weather?")
        
        # Should call retrieval service
        assert mock_dependencies['retrieval_instance'].retrieve.called
        
        # Should call LLM
        assert mock_dependencies['llm_instance'].complete.called
        
        # Should return response
        assert response == "Test response"
    
    def test_chat_with_custom_n_results(self, mock_dependencies):
        """Test chat with custom number of retrieval results."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        bot.chat("Test query", n_results=5)
        
        # Should call retrieve with n_results=5
        mock_dependencies['retrieval_instance'].retrieve.assert_called_with("Test query", n_results=5)
    
    def test_chat_calls_prompt_engine(self, mock_dependencies):
        """Test that chat() calls prompt engine with correct parameters."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        bot.chat("Test query")
        
        # Should call construct_prompt
        assert mock_dependencies['prompt_instance'].construct_prompt.called
        
        # Get the call arguments
        call_args = mock_dependencies['prompt_instance'].construct_prompt.call_args
        
        # Should pass context_chunks, chat_history, and user_task
        assert 'context_chunks' in call_args.kwargs or len(call_args.args) > 0
        assert 'user_task' in call_args.kwargs or len(call_args.args) > 2


class TestChatHistory:
    """Tests for chat history accumulation."""
    
    def test_chat_history_accumulation(self, mock_dependencies):
        """Test that chat history accumulates correctly."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Initial state
        assert len(bot.chat_history) == 0
        
        # First message
        bot.chat("First message")
        assert len(bot.chat_history) == 2  # user + assistant
        assert bot.chat_history[0]['role'] == 'user'
        assert bot.chat_history[0]['content'] == "First message"
        assert bot.chat_history[1]['role'] == 'assistant'
        
        # Second message
        bot.chat("Second message")
        assert len(bot.chat_history) == 4  # 2 previous + 2 new
        assert bot.chat_history[2]['role'] == 'user'
        assert bot.chat_history[2]['content'] == "Second message"
    
    def test_chat_history_format(self, mock_dependencies):
        """Test that chat history is stored in correct format."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        bot.chat("Test message")
        
        # Check user message format
        assert bot.chat_history[0] == {
            'role': 'user',
            'content': 'Test message'
        }
        
        # Check assistant message format
        assert bot.chat_history[1]['role'] == 'assistant'
        assert 'content' in bot.chat_history[1]


class TestChatAutoReset:
    """Tests for automatic context reset on token limit."""
    
    def test_chat_auto_reset_on_token_limit(self, mock_dependencies):
        """Test that chat auto-resets when token limit is reached."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 100
        
        # Add some history
        bot.chat_history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ]
        
        # Mock count_tokens to return value >= max_context_tokens
        mock_dependencies['llm_instance'].count_tokens.return_value = 100
        
        response = bot.chat("New message")
        
        # History should have been reset and new message added
        # After reset: 0 messages, after chat: 2 messages (user + assistant)
        assert len(bot.chat_history) == 2
        assert bot.chat_history[0]['content'] == "New message"
    
    def test_chat_auto_reset_warning_message(self, mock_dependencies):
        """Test that auto-reset includes warning message."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=1)
        bot.max_context_tokens = 100
        
        bot.chat_history = [{"role": "user", "content": "Old"}]
        mock_dependencies['llm_instance'].count_tokens.return_value = 100
        
        response = bot.chat("New message")
        
        # Response should include auto-reset warning
        assert "⚠️" in response or "автоматически сброшен" in response
    
    def test_chat_no_auto_reset_below_limit(self, mock_dependencies):
        """Test that auto-reset doesn't happen below token limit."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        
        bot.chat_history = [{"role": "user", "content": "Previous"}]
        mock_dependencies['llm_instance'].count_tokens.return_value = 50
        
        bot.chat("New message")
        
        # History should include both old and new messages
        assert len(bot.chat_history) == 3  # 1 old user + 1 new user + 1 new assistant


class TestChatErrorHandling:
    """Tests for error handling in chat()."""
    
    def test_chat_llm_failure(self, mock_dependencies):
        """Test chat behavior when LLM fails."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Make LLM raise an exception
        mock_dependencies['llm_instance'].complete.side_effect = Exception("LLM Error")
        
        # Chat should handle the error gracefully
        with pytest.raises(Exception):
            bot.chat("Test query")
    
    def test_chat_retrieval_failure(self, mock_dependencies):
        """Test chat behavior when retrieval fails."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Make retrieval raise an exception
        mock_dependencies['retrieval_instance'].retrieve.side_effect = Exception("Retrieval Error")
        
        # Chat should handle the error gracefully
        with pytest.raises(Exception):
            bot.chat("Test query")
    
    def test_chat_empty_retrieval_results(self, mock_dependencies):
        """Test chat with empty retrieval results."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Return empty list from retrieval
        mock_dependencies['retrieval_instance'].retrieve.return_value = []
        
        response = bot.chat("Test query")
        
        # Should still work and return response
        assert response == "Test response"


class TestChatIntegration:
    """Integration tests for chat flow."""
    
    def test_chat_full_flow(self, mock_dependencies):
        """Test complete chat flow from query to response."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Mock the full flow
        mock_dependencies['retrieval_instance'].retrieve.return_value = ["Context 1", "Context 2"]
        mock_dependencies['prompt_instance'].construct_prompt.return_value = "Full prompt"
        mock_dependencies['llm_instance'].complete.return_value = "Final response"
        
        response = bot.chat("What is the answer?")
        
        # Verify the flow
        # 1. Retrieval was called
        mock_dependencies['retrieval_instance'].retrieve.assert_called_once()
        
        # 2. Prompt was constructed
        mock_dependencies['prompt_instance'].construct_prompt.assert_called_once()
        
        # 3. LLM was called
        mock_dependencies['llm_instance'].complete.assert_called_once()
        
        # 4. Response was returned
        assert response == "Final response"
        
        # 5. History was updated
        assert len(bot.chat_history) == 2
        assert bot.chat_history[0]['content'] == "What is the answer?"
        assert bot.chat_history[1]['content'] == "Final response"
    
    def test_chat_conversation_flow(self, mock_dependencies):
        """Test multi-turn conversation."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # First turn
        response1 = bot.chat("First question")
        assert len(bot.chat_history) == 2
        
        # Second turn
        response2 = bot.chat("Second question")
        assert len(bot.chat_history) == 4
        
        # Third turn
        response3 = bot.chat("Third question")
        assert len(bot.chat_history) == 6
        
        # Verify all messages are in history
        assert bot.chat_history[0]['content'] == "First question"
        assert bot.chat_history[2]['content'] == "Second question"
        assert bot.chat_history[4]['content'] == "Third question"
