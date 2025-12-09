"""
Tests for LLM Client functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.llm import LLMClient


class TestLLMInit:
    """Tests for LLMClient initialization."""
    
    def test_init_with_openrouter_api_key(self):
        """Test initialization with OPENROUTER_API_KEY."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI') as mock_openai:
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                
                assert client.api_key == 'test-key'
                assert client.model == 'test-model'
                assert mock_openai.called
    
    def test_init_with_openai_api_key(self):
        """Test initialization with OPENAI_API_KEY (fallback)."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'openai-key'}, clear=True):
            with patch('src.core.llm.OpenAI') as mock_openai:
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                
                assert client.api_key == 'openai-key'
    
    def test_init_missing_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API_KEY"):
                LLMClient(model="test-model")
    
    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_BASE_URL': 'https://custom.api.com/v1'
        }):
            with patch('src.core.llm.OpenAI') as mock_openai:
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                
                assert client.base_url == 'https://custom.api.com/v1'
                mock_openai.assert_called_with(
                    base_url='https://custom.api.com/v1',
                    api_key='test-key'
                )
    
    def test_init_default_base_url(self):
        """Test initialization with default base URL."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI'):
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                
                assert client.base_url == 'https://openrouter.ai/api/v1'
    
    def test_init_tokenizer_known_model(self):
        """Test tokenizer initialization with known model."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI'):
                with patch('src.core.llm.tiktoken.encoding_for_model') as mock_encoding:
                    mock_encoding.return_value = Mock()
                    
                    client = LLMClient(model="gpt-3.5-turbo", log_level=LOG_WARNING)
                    
                    # Should try to get encoding for the model name
                    mock_encoding.assert_called_with('gpt-3.5-turbo')
    
    def test_init_tokenizer_fallback(self):
        """Test tokenizer fallback for unknown model."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI'):
                with patch('src.core.llm.tiktoken.encoding_for_model') as mock_encoding_for:
                    with patch('src.core.llm.tiktoken.get_encoding') as mock_get_encoding:
                        # Simulate KeyError for unknown model
                        mock_encoding_for.side_effect = KeyError("Unknown model")
                        mock_get_encoding.return_value = Mock()
                        
                        client = LLMClient(model="unknown/model", log_level=LOG_WARNING)
                        
                        # Should fallback to cl100k_base
                        mock_get_encoding.assert_called_with('cl100k_base')
    
    def test_init_log_levels(self):
        """Test initialization with different log levels."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI'):
                for log_level in [LOG_WARNING, LOG_INFO, LOG_DEBUG]:
                    client = LLMClient(model="test-model", log_level=log_level)
                    assert client.log_level == log_level


class TestTokenCounting:
    """Tests for count_tokens() method."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client for testing."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI'):
                with patch('src.core.llm.tiktoken.get_encoding') as mock_encoding:
                    mock_enc = Mock()
                    mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens
                    mock_encoding.return_value = mock_enc
                    
                    client = LLMClient(model="test-model", log_level=LOG_WARNING)
                    yield client
    
    def test_count_tokens_empty_messages(self, llm_client):
        """Test counting tokens for empty message list."""
        count = llm_client.count_tokens([])
        
        # Should return 2 (reply priming tokens)
        assert count == 2
    
    def test_count_tokens_single_message(self, llm_client):
        """Test counting tokens for single message."""
        messages = [{"role": "user", "content": "Hello"}]
        
        count = llm_client.count_tokens(messages)
        
        # 4 (message overhead) + 3 (role tokens) + 3 (content tokens) + 2 (reply priming) = 12
        assert count > 0
    
    def test_count_tokens_multiple_messages(self, llm_client):
        """Test counting tokens for multiple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        count = llm_client.count_tokens(messages)
        
        # Should count all messages
        assert count > 0
    
    def test_count_tokens_long_messages(self, llm_client):
        """Test counting tokens for long messages."""
        long_content = "This is a very long message " * 100
        messages = [{"role": "user", "content": long_content}]
        
        count = llm_client.count_tokens(messages)
        
        # Should handle long messages
        assert count > 0
    
    def test_count_tokens_special_characters(self, llm_client):
        """Test counting tokens with special characters."""
        messages = [
            {"role": "user", "content": "Hello! 你好 @#$%"}
        ]
        
        count = llm_client.count_tokens(messages)
        
        # Should handle special characters
        assert count > 0


class TestCompletion:
    """Tests for complete() method."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client with mocked OpenAI."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                client.client = mock_client
                
                yield client, mock_client
    
    def test_complete_successful_response(self, llm_client):
        """Test successful completion."""
        client, mock_client = llm_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client.complete(messages)
        
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_complete_with_custom_temperature(self, llm_client):
        """Test completion with custom temperature."""
        client, mock_client = llm_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        client.complete(messages, temperature=0.5)
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['temperature'] == 0.5
    
    def test_complete_with_custom_max_tokens(self, llm_client):
        """Test completion with custom max_tokens."""
        client, mock_client = llm_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        client.complete(messages, max_tokens=1000)
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['max_tokens'] == 1000
    
    def test_complete_empty_response(self, llm_client):
        """Test handling of empty response."""
        client, mock_client = llm_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        response = client.complete(messages)
        
        assert response == ""
    
    def test_complete_api_error(self, llm_client):
        """Test handling of API error."""
        client, mock_client = llm_client
        
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Should raise exception now
        with pytest.raises(Exception, match="API Error"):
             client.complete(messages)

    def test_complete_network_timeout(self, llm_client):
        """Test handling of network timeout."""
        client, mock_client = llm_client
        
        mock_client.chat.completions.create.side_effect = TimeoutError("Timeout")
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Should raise exception now
        with pytest.raises(TimeoutError, match="Timeout"):
             client.complete(messages)
    
    def test_complete_log_level_warning(self, llm_client):
        """Test completion with LOG_WARNING (no output)."""
        client, mock_client = llm_client
        client.log_level = LOG_WARNING
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Should not print anything (we can't easily test this, but at least it shouldn't crash)
        response = client.complete(messages)
        assert response == "Response"
    
    def test_complete_log_level_debug(self, llm_client, capsys):
        """Test completion with LOG_DEBUG (full logging)."""
        client, mock_client = llm_client
        client.log_level = LOG_DEBUG
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        response = client.complete(messages)
        
        # With log_level=LOG_DEBUG, debug info goes to syslog2, not stdout
        # Just verify the call succeeded
        assert response == "Response"


class TestStreaming:
    """Tests for stream_complete() method."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client with mocked OpenAI."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            with patch('src.core.llm.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                client = LLMClient(model="test-model", log_level=LOG_WARNING)
                client.client = mock_client
                
                yield client, mock_client
    
    def test_stream_complete_successful(self, llm_client):
        """Test successful streaming completion."""
        client, mock_client = llm_client
        
        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" "))]),
            Mock(choices=[Mock(delta=Mock(content="world"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [{"role": "user", "content": "Test"}]
        chunks = list(client.stream_complete(messages))
        
        assert chunks == ["Hello", " ", "world"]
    
    def test_stream_complete_chunk_assembly(self, llm_client):
        """Test that chunks can be assembled into full response."""
        client, mock_client = llm_client
        
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="This"))]),
            Mock(choices=[Mock(delta=Mock(content=" is"))]),
            Mock(choices=[Mock(delta=Mock(content=" a"))]),
            Mock(choices=[Mock(delta=Mock(content=" test"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [{"role": "user", "content": "Test"}]
        full_response = "".join(client.stream_complete(messages))
        
        assert full_response == "This is a test"
    
    def test_stream_complete_error_during_streaming(self, llm_client):
        """Test error handling during streaming."""
        client, mock_client = llm_client
        
        def error_generator():
            yield Mock(choices=[Mock(delta=Mock(content="Start"))])
            raise Exception("Streaming error")
        
        mock_client.chat.completions.create.return_value = error_generator()
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception):
            list(client.stream_complete(messages))
    
    def test_stream_complete_none_content(self, llm_client):
        """Test streaming with None content (should be filtered)."""
        client, mock_client = llm_client
        
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # None content
            Mock(choices=[Mock(delta=Mock(content="world"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        
        messages = [{"role": "user", "content": "Test"}]
        chunks = list(client.stream_complete(messages))
        
        # None content should be filtered out
        assert chunks == ["Hello", "world"]
