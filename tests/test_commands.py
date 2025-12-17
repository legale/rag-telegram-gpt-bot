"""Tests for src/core/commands.py"""

import pytest
from unittest.mock import Mock, patch
from src.core.commands import (
    StartCommandHandler,
    HelpCommandHandler,
    ResetCommandHandler,
    TokensCommandHandler,
    ModelCommandHandler,
    FindCommandHandler
)
from src.core.dispatcher import CommandContext, CommandResult


class TestStartCommandHandler:
    """Tests for StartCommandHandler"""
    
    def test_handle_success(self):
        """Test successful /start command handling"""
        handler = StartCommandHandler()
        context = CommandContext(command_name="start")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert "Привет!" in result.message
        assert "/help" in result.message


class TestHelpCommandHandler:
    """Tests for HelpCommandHandler"""
    
    def test_handle_success(self):
        """Test successful /help command handling"""
        handler = HelpCommandHandler()
        context = CommandContext(command_name="help")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert "Доступные команды" in result.message
        assert "/start" in result.message
        assert "/help" in result.message
        assert "/reset" in result.message
        assert "/tokens" in result.message
        assert "/model" in result.message
        assert "/find" in result.message


class TestResetCommandHandler:
    """Tests for ResetCommandHandler"""
    
    def test_init(self):
        """Test ResetCommandHandler initialization"""
        mock_bot = Mock()
        handler = ResetCommandHandler(mock_bot)
        
        assert handler.bot == mock_bot
    
    def test_handle_success(self):
        """Test successful /reset command handling"""
        mock_bot = Mock()
        mock_bot.reset_context.return_value = "Контекст сброшен"
        handler = ResetCommandHandler(mock_bot)
        context = CommandContext(command_name="reset")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert result.message == "Контекст сброшен"
        mock_bot.reset_context.assert_called_once()
    
    def test_handle_exception(self):
        """Test /reset command handling with exception"""
        mock_bot = Mock()
        mock_bot.reset_context.side_effect = Exception("Reset failed")
        handler = ResetCommandHandler(mock_bot)
        context = CommandContext(command_name="reset")
        
        with patch('src.lib.syslog2.syslog2') as mock_syslog:
            result = handler.handle(context)
            
            assert result.success is False
            assert "Ошибка при сбросе контекста" in result.message
            assert result.error == "Reset failed"
            mock_syslog.assert_called_once()


class TestTokensCommandHandler:
    """Tests for TokensCommandHandler"""
    
    def test_init(self):
        """Test TokensCommandHandler initialization"""
        mock_bot = Mock()
        handler = TokensCommandHandler(mock_bot)
        
        assert handler.bot == mock_bot
    
    def test_handle_success_low_usage(self):
        """Test successful /tokens command with low usage"""
        mock_bot = Mock()
        mock_bot.get_token_usage.return_value = {
            'current_tokens': 100,
            'max_tokens': 1000,
            'percentage': 10
        }
        handler = TokensCommandHandler(mock_bot)
        context = CommandContext(command_name="tokens")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert "Использование токенов" in result.message
        assert "Достаточно места" in result.message
        assert result.data['current_tokens'] == 100
        assert result.data['max_tokens'] == 1000
        assert result.data['percentage'] == 10
    
    def test_handle_success_medium_usage(self):
        """Test successful /tokens command with medium usage"""
        mock_bot = Mock()
        mock_bot.get_token_usage.return_value = {
            'current_tokens': 600,
            'max_tokens': 1000,
            'percentage': 60
        }
        handler = TokensCommandHandler(mock_bot)
        context = CommandContext(command_name="tokens")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert "Контекст заполнен наполовину" in result.message
    
    def test_handle_success_high_usage(self):
        """Test successful /tokens command with high usage"""
        mock_bot = Mock()
        mock_bot.get_token_usage.return_value = {
            'current_tokens': 850,
            'max_tokens': 1000,
            'percentage': 85
        }
        handler = TokensCommandHandler(mock_bot)
        context = CommandContext(command_name="tokens")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert "Приближаетесь к лимиту" in result.message
        assert "/reset" in result.message
    
    def test_handle_exception(self):
        """Test /tokens command handling with exception"""
        mock_bot = Mock()
        mock_bot.get_token_usage.side_effect = Exception("Token usage failed")
        handler = TokensCommandHandler(mock_bot)
        context = CommandContext(command_name="tokens")
        
        with patch('src.lib.syslog2.syslog2') as mock_syslog:
            result = handler.handle(context)
            
            assert result.success is False
            assert "Ошибка при получении информации о токенах" in result.message
            assert result.error == "Token usage failed"
            mock_syslog.assert_called_once()


class TestModelCommandHandler:
    """Tests for ModelCommandHandler"""
    
    def test_init_without_admin_manager(self):
        """Test ModelCommandHandler initialization without admin_manager"""
        mock_bot = Mock()
        handler = ModelCommandHandler(mock_bot)
        
        assert handler.bot == mock_bot
        assert handler.admin_manager is None
    
    def test_init_with_admin_manager(self):
        """Test ModelCommandHandler initialization with admin_manager"""
        mock_bot = Mock()
        mock_admin = Mock()
        handler = ModelCommandHandler(mock_bot, mock_admin)
        
        assert handler.bot == mock_bot
        assert handler.admin_manager == mock_admin
    
    def test_handle_success_without_admin_manager(self):
        """Test successful /model command without admin_manager"""
        mock_bot = Mock()
        mock_bot.get_model.return_value = "Модель: gpt-4"
        mock_bot.current_model_name = "gpt-4"
        handler = ModelCommandHandler(mock_bot)
        context = CommandContext(command_name="model")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert result.message == "Модель: gpt-4"
        assert result.data['model'] == "gpt-4"
        mock_bot.get_model.assert_called_once()
    
    def test_handle_success_with_admin_manager(self):
        """Test successful /model command with admin_manager"""
        mock_bot = Mock()
        mock_bot.get_model.return_value = "Модель: gpt-4"
        mock_bot.current_model_name = "gpt-4"
        mock_admin = Mock()
        mock_admin.config = Mock()
        handler = ModelCommandHandler(mock_bot, mock_admin)
        context = CommandContext(command_name="model")
        
        result = handler.handle(context)
        
        assert result.success is True
        assert mock_admin.config.current_model == "gpt-4"
    
    def test_handle_success_with_admin_manager_config_error(self):
        """Test /model command when config save fails"""
        mock_bot = Mock()
        mock_bot.get_model.return_value = "Модель: gpt-4"
        mock_bot.current_model_name = "gpt-4"
        mock_admin = Mock()
        mock_admin.config = Mock()
        def set_model_side_effect(value):
            if value == "gpt-4":
                raise Exception("Config error")
        mock_admin.config.current_model = "old-model"
        type(mock_admin.config).current_model = property(
            lambda self: "old-model",
            set_model_side_effect
        )
        handler = ModelCommandHandler(mock_bot, mock_admin)
        context = CommandContext(command_name="model")
        
        with patch('src.lib.syslog2.syslog2') as mock_syslog:
            result = handler.handle(context)
            
            assert result.success is True
            mock_syslog.assert_called()
    
    def test_handle_exception(self):
        """Test /model command handling with exception"""
        mock_bot = Mock()
        mock_bot.get_model.side_effect = Exception("Model switch failed")
        handler = ModelCommandHandler(mock_bot)
        context = CommandContext(command_name="model")
        
        with patch('src.lib.syslog2.syslog2') as mock_syslog:
            result = handler.handle(context)
            
            assert result.success is False
            assert "Ошибка при переключении модели" in result.message
            assert result.error == "Model switch failed"
            mock_syslog.assert_called_once()


class TestFindCommandHandler:
    """Tests for FindCommandHandler"""
    
    def test_init_without_admin_manager(self):
        """Test FindCommandHandler initialization without admin_manager"""
        mock_bot = Mock()
        handler = FindCommandHandler(mock_bot, debug_rag=False)
        
        assert handler.bot == mock_bot
        assert handler.admin_manager is None
        assert handler.debug_rag is False
    
    def test_init_with_admin_manager(self):
        """Test FindCommandHandler initialization with admin_manager"""
        mock_bot = Mock()
        mock_admin = Mock()
        handler = FindCommandHandler(mock_bot, mock_admin, debug_rag=True)
        
        assert handler.bot == mock_bot
        assert handler.admin_manager == mock_admin
        assert handler.debug_rag is True
    
    def test_handle_invalid_args(self):
        """Test /find command with invalid arguments"""
        mock_bot = Mock()
        handler = FindCommandHandler(mock_bot)
        context = CommandContext(command_name="find", args=[])
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse:
            mock_parse.return_value = (None, None, "Invalid arguments")
            result = handler.handle(context)
            
            assert result.success is False
            assert result.message == "Invalid arguments"
            assert result.error == "Invalid arguments"
    
    def test_handle_list_action(self):
        """Test /find command with list action"""
        mock_bot = Mock()
        handler = FindCommandHandler(mock_bot)
        context = CommandContext(command_name="find", args=["hybrid", "list"])
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse:
            mock_parse.return_value = ("hybrid", "list", None)
            result = handler.handle(context)
            
            assert result.success is True
            assert "Доступные методы RAG" in result.message
            assert "hybrid" in result.message
            assert "vector_only" in result.message
            assert "fts_only" in result.message
            assert result.data['rag_method'] == "hybrid"
            assert result.data['action'] == "list"
    
    def test_handle_search_no_results(self):
        """Test /find command with search that returns no results"""
        mock_bot = Mock()
        mock_bot.db = Mock()
        mock_bot.db.db_url = "sqlite:///test.db"
        mock_bot.vector_store = Mock()
        mock_bot.vector_store.persist_directory = "/tmp/vec"
        mock_bot.embedding_client = Mock()
        mock_bot.log_level = 4
        handler = FindCommandHandler(mock_bot)
        context = CommandContext(command_name="find", args=["hybrid", "test", "query"])
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse, \
             patch('src.app.bootstrap.create_hybrid_retrieval') as mock_create, \
             patch('src.core.message_search.search_message_contents') as mock_search:
            mock_parse.return_value = ("hybrid", None, "test query")
            mock_retrieval = Mock()
            mock_create.return_value = mock_retrieval
            mock_search.return_value = []
            
            result = handler.handle(context)
            
            assert result.success is True
            assert "ничего не найдено" in result.message
            assert result.data['results_count'] == 0
            assert result.data['rag_method'] == "hybrid"
    
    def test_handle_search_with_results(self):
        """Test /find command with search that returns results"""
        mock_bot = Mock()
        mock_bot.db = Mock()
        mock_bot.db.db_url = "sqlite:///test.db"
        mock_bot.vector_store = Mock()
        mock_bot.vector_store.persist_directory = "/tmp/vec"
        mock_bot.embedding_client = Mock()
        mock_bot.log_level = 4
        handler = FindCommandHandler(mock_bot)
        context = CommandContext(command_name="find", args=["hybrid", "test", "query"])
        
        mock_message_parts = [[{"text": "result1"}], [{"text": "result2"}]]
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse, \
             patch('src.app.bootstrap.create_hybrid_retrieval') as mock_create, \
             patch('src.core.message_search.search_message_contents') as mock_search:
            mock_parse.return_value = ("hybrid", None, "test query")
            mock_retrieval = Mock()
            mock_create.return_value = mock_retrieval
            mock_search.return_value = mock_message_parts
            
            result = handler.handle(context)
            
            assert result.success is True
            assert result.message == ""
            assert result.data['results_count'] == 2
            assert result.data['rag_method'] == "hybrid"
            assert result.data['needs_formatting'] is True
            assert result.data['message_parts_list'] == mock_message_parts
    
    def test_handle_search_with_admin_manager_threshold(self):
        """Test /find command with admin_manager providing threshold"""
        mock_bot = Mock()
        mock_bot.db = Mock()
        mock_bot.db.db_url = "sqlite:///test.db"
        mock_bot.vector_store = Mock()
        mock_bot.vector_store.persist_directory = "/tmp/vec"
        mock_bot.embedding_client = Mock()
        mock_bot.log_level = 4
        mock_admin = Mock()
        mock_admin.config = Mock()
        mock_admin.config.cosine_distance_thr = 2.0
        handler = FindCommandHandler(mock_bot, mock_admin)
        context = CommandContext(command_name="find", args=["hybrid", "test", "query"])
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse, \
             patch('src.app.bootstrap.create_hybrid_retrieval') as mock_create, \
             patch('src.core.message_search.search_message_contents') as mock_search:
            mock_parse.return_value = ("hybrid", None, "test query")
            mock_retrieval = Mock()
            mock_create.return_value = mock_retrieval
            mock_search.return_value = []
            
            result = handler.handle(context)
            
            # Verify threshold was passed to search_message_contents
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[1]['threshold'] == 2.0
    
    def test_handle_exception(self):
        """Test /find command handling with exception"""
        mock_bot = Mock()
        handler = FindCommandHandler(mock_bot)
        context = CommandContext(command_name="find", args=["hybrid", "test"])
        
        with patch('src.bot.command_parser.parse_find_command_args') as mock_parse, \
             patch('src.lib.syslog2.syslog2') as mock_syslog:
            mock_parse.side_effect = Exception("Search failed")
            result = handler.handle(context)
            
            assert result.success is False
            assert "Ошибка при выполнении поиска" in result.message
            assert result.error == "Search failed"
            mock_syslog.assert_called_once()
