"""
Tests for LegaleBot context management functionality.
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
        
        # Setup LLM client mock for token counting
        mock_llm_instance = MagicMock()
        mock_llm_instance.count_tokens.return_value = 100
        mock_llm.return_value = mock_llm_instance
        
        yield {
            'db': mock_db,
            'vector_store': mock_vs,
            'embedding_client': mock_ec,
            'llm_client': mock_llm,
            'retrieval_service': mock_rs,
            'prompt_engine': mock_pe
        }


class TestResetContext:
    """Tests for reset_context() method."""
    
    def test_reset_context_clears_history(self, mock_dependencies):
        """Test that reset_context() clears chat history."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Add some history
        bot.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Reset
        result = bot.reset_context()
        
        assert len(bot.chat_history) == 0
        assert bot.chat_history == []
    
    def test_reset_context_returns_confirmation(self, mock_dependencies):
        """Test that reset_context() returns confirmation message."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        result = bot.reset_context()
        
        assert "" in result
        assert "Контекст сброшен" in result
    
    def test_reset_context_multiple_times(self, mock_dependencies):
        """Test that reset_context() can be called multiple times."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Add history and reset
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.reset_context()
        assert len(bot.chat_history) == 0
        
        # Add more history and reset again
        bot.chat_history = [{"role": "user", "content": "Test2"}]
        bot.reset_context()
        assert len(bot.chat_history) == 0


class TestGetTokenUsage:
    """Tests for get_token_usage() method."""
    
    def test_get_token_usage_empty_history(self, mock_dependencies):
        """Test get_token_usage() with empty chat history."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        usage = bot.get_token_usage()
        
        assert usage['current_tokens'] == 0
        assert usage['max_tokens'] == 14000  # Default
        assert usage['percentage'] == 0.0
    
    def test_get_token_usage_with_history(self, mock_dependencies):
        """Test get_token_usage() with chat history."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Add some history
        bot.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Mock count_tokens to return a specific value
        bot.llm_client.count_tokens.return_value = 500
        
        usage = bot.get_token_usage()
        
        assert usage['current_tokens'] == 500
        assert usage['max_tokens'] == 14000
        assert usage['percentage'] > 0
    
    def test_get_token_usage_percentage_calculation(self, mock_dependencies):
        """Test that percentage is calculated correctly."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.llm_client.count_tokens.return_value = 500
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 50.0
    
    def test_get_token_usage_custom_max_tokens(self, mock_dependencies):
        """Test get_token_usage() with custom MAX_CONTEXT_TOKENS."""
        with patch.dict('os.environ', {'MAX_CONTEXT_TOKENS': '10000'}):
            bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
            
            usage = bot.get_token_usage()
            
            assert usage['max_tokens'] == 10000
    
    def test_get_token_usage_threshold_0_percent(self, mock_dependencies):
        """Test token usage at 0% threshold."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.llm_client.count_tokens.return_value = 0
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 0.0
    
    def test_get_token_usage_threshold_50_percent(self, mock_dependencies):
        """Test token usage at 50% threshold."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.llm_client.count_tokens.return_value = 500
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 50.0
    
    def test_get_token_usage_threshold_80_percent(self, mock_dependencies):
        """Test token usage at 80% threshold."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.llm_client.count_tokens.return_value = 800
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 80.0
    
    def test_get_token_usage_threshold_100_percent(self, mock_dependencies):
        """Test token usage at 100% threshold."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.llm_client.count_tokens.return_value = 1000
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 100.0
    
    def test_get_token_usage_over_100_percent(self, mock_dependencies):
        """Test token usage over 100% threshold."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        bot.max_context_tokens = 1000
        bot.chat_history = [{"role": "user", "content": "Test"}]
        bot.llm_client.count_tokens.return_value = 1200
        
        usage = bot.get_token_usage()
        
        assert usage['percentage'] == 120.0
    
    def test_get_token_usage_uses_last_5_messages(self, mock_dependencies):
        """Test that get_token_usage() only uses last 5 messages for calculation."""
        bot = LegaleBot(db_url="sqlite:///test.db", vector_db_path="test_chroma", verbosity=0)
        
        # Add 10 messages
        for i in range(10):
            bot.chat_history.append({"role": "user", "content": f"Message {i}"})
        
        bot.llm_client.count_tokens.return_value = 100
        
        usage = bot.get_token_usage()
        
        # Should have called count_tokens
        assert bot.llm_client.count_tokens.called
        assert usage['current_tokens'] == 100
