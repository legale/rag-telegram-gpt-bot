"""
Tests for Telegram bot webhook and message handling.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
from telegram import Update, Message, User, Chat
from telegram.ext import Application
from fastapi import Request, Response
from types import SimpleNamespace

from src.bot.tgbot import (
    handle_start_command,
    handle_help_command,
    handle_reset_command,
    handle_tokens_command,
    handle_model_command,
    handle_admin_set_command,
    handle_admin_get_command,
    handle_admin_command,
    handle_find_command,
    handle_user_query,
    route_command,
    _register_command_group,
    _get_profile_paths,
    _create_admin_manager,
    _map_log_level_to_constant,
    _get_bot_configuration,
    _create_legale_bot,
    _register_admin_commands,
    reload_for_current_profile,
    _map_syslog2_to_logging_level,
    setup_logging,
    lifespan,
    process_document_update,
    process_text_update,
    _parse_webhook_update,
    _process_webhook_update,
    _setup_webhook_endpoint,
    create_app,
    is_bot_mentioned,
    _check_access,
    _extract_mention_text,
    _parse_search_command,
    _parse_search_mention,
    _send_message_parts_unified,
    _send_message_parts,
    _send_search_results,
    _should_ignore_message,
    _handle_search_mention,
    _process_command_message,
    _process_regular_message,
    _determine_response_decision,
    _send_response_if_available,
    _handle_public_commands_step,
    _check_access_step,
    get_runtime_context,
    _determine_response_step,
    _handle_search_mention_step,
    _route_message_step,
    handle_message,
    _handle_public_commands
)


class TestMessageHandling:
    @pytest.fixture
    def mock_bot(self):
        """Mock LegaleBot instance."""
        bot = Mock()
        bot.reset_context.return_value = "Context reset"
        bot.get_token_usage.return_value = {
            "current_tokens": 100,
            "max_tokens": 1000,
            "percentage": 10
        }
        bot.get_model.return_value = "Model: gpt-4"
        bot.current_model_name = "gpt-4"
        bot.chat.return_value = "Response"
        bot.get_rag_debug_info.return_value = {
            "chunks": [],
            "prompt": "System prompt",
            "token_count": 100
        }
        bot.db = Mock()
        bot.embedding_client = Mock()
        bot.retrieval = Mock()
        return bot

    @pytest.fixture
    def mock_admin_manager(self):
        """Mock AdminManager instance."""
        manager = Mock()
        manager.config = Mock()
        manager.config.current_model = "gpt-4"
        manager.config.get_system_prompt.return_value = "System prompt"
        manager.verify_password.return_value = False
        manager.is_admin.return_value = False
        manager.get_admin.return_value = None
        manager.set_admin.return_value = None
        return manager

    @pytest.fixture
    def mock_admin_router(self):
        """Mock AdminCommandRouter instance."""
        router = Mock()
        router.route = AsyncMock(return_value="Admin response")
        return router

    @pytest.fixture(autouse=True)
    def setup_globals(self, mock_bot, mock_admin_manager, mock_admin_router):
        """Setup global variables for tests."""
        with patch("src.bot.tgbot._bot_instance", mock_bot), \
             patch("src.bot.tgbot._admin_manager", mock_admin_manager), \
             patch("src.bot.tgbot._admin_router", mock_admin_router):
            yield

    @pytest.mark.asyncio
    async def test_handle_start_command(self):
        """Test handling /start command."""
        result = await handle_start_command()
        assert "Привет" in result
        assert "/help" in result

    @pytest.mark.asyncio
    async def test_handle_help_command(self):
        """Test handling /help command."""
        result = await handle_help_command()
        assert "команды" in result or "команд" in result
        assert "/start" in result

    @pytest.mark.asyncio
    async def test_handle_reset_command(self, mock_bot):
        """Test handling /reset command."""
        result = await handle_reset_command()
        assert result == "Context reset"
        mock_bot.reset_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_reset_command_error(self, mock_bot):
        """Test handling /reset command with error."""
        mock_bot.reset_context.side_effect = Exception("Error")
        result = await handle_reset_command()
        assert "Ошибка" in result

    @pytest.mark.asyncio
    async def test_handle_tokens_command(self, mock_bot):
        """Test handling /tokens command."""
        result = await handle_tokens_command()
        assert "токенов" in result or "Текущее" in result
        mock_bot.get_token_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tokens_command_error(self, mock_bot):
        """Test handling /tokens command with error."""
        mock_bot.get_token_usage.side_effect = Exception("Error")
        result = await handle_tokens_command()
        assert "Ошибка" in result

    @pytest.mark.asyncio
    async def test_handle_model_command(self, mock_bot, mock_admin_manager):
        """Test handling /model command."""
        result = await handle_model_command()
        assert "Model" in result or "модель" in result
        mock_bot.get_model.assert_called_once()
        assert mock_admin_manager.config.current_model == "gpt-4"

    @pytest.mark.asyncio
    async def test_handle_model_command_error(self, mock_bot):
        """Test handling /model command with error."""
        mock_bot.get_model.side_effect = Exception("Error")
        result = await handle_model_command()
        assert "Ошибка" in result

    @pytest.mark.asyncio
    async def test_handle_admin_set_command_no_manager(self):
        """Test handling /admin_set without admin manager."""
        with patch("src.bot.tgbot._admin_manager", None):
            message = Mock()
            result = await handle_admin_set_command("/admin_set password", message)
            assert "недоступна" in result

    @pytest.mark.asyncio
    async def test_handle_admin_set_command_invalid_format(self):
        """Test handling /admin_set with invalid format."""
        message = Mock()
        result = await handle_admin_set_command("/admin_set", message)
        assert "формат" in result or "Использование" in result

    @pytest.mark.asyncio
    async def test_handle_admin_set_command_success(self, mock_admin_manager):
        """Test handling /admin_set with correct password."""
        mock_admin_manager.verify_password.return_value = True
        message = Mock()
        message.from_user = Mock()
        message.from_user.id = 123
        message.from_user.username = "testuser"
        message.from_user.first_name = "Test"
        message.from_user.last_name = "User"
        
        result = await handle_admin_set_command("/admin_set password", message)
        assert "администратором" in result or "успешно" in result
        mock_admin_manager.set_admin.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_admin_set_command_wrong_password(self, mock_admin_manager):
        """Test handling /admin_set with wrong password."""
        mock_admin_manager.verify_password.return_value = False
        message = Mock()
        message.from_user = Mock()
        message.from_user.id = 123
        
        result = await handle_admin_set_command("/admin_set wrong", message)
        assert "Неверный пароль" in result or "пароль" in result

    @pytest.mark.asyncio
    async def test_handle_admin_get_command_no_manager(self):
        """Test handling /admin_get without admin manager."""
        with patch("src.bot.tgbot._admin_manager", None):
            result = await handle_admin_get_command(123)
            assert "недоступна" in result

    @pytest.mark.asyncio
    async def test_handle_admin_get_command_not_admin(self, mock_admin_manager):
        """Test handling /admin_get when user is not admin."""
        mock_admin_manager.is_admin.return_value = False
        result = await handle_admin_get_command(123)
        assert "администратор" in result or "доступна" in result

    @pytest.mark.asyncio
    async def test_handle_admin_get_command_success(self, mock_admin_manager):
        """Test handling /admin_get when user is admin."""
        mock_admin_manager.is_admin.return_value = True
        mock_admin_manager.get_admin.return_value = {
            "full_name": "Test User",
            "user_id": 123,
            "username": "testuser"
        }
        result = await handle_admin_get_command(123)
        assert "Администратор" in result or "администратор" in result

    @pytest.mark.asyncio
    async def test_handle_admin_command(self, mock_admin_router):
        """Test handling /admin command."""
        update = Mock()
        result = await handle_admin_command(update)
        assert result == "Admin response"
        mock_admin_router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_admin_command_no_router(self):
        """Test handling /admin command without router."""
        with patch("src.bot.tgbot._admin_router", None):
            update = Mock()
            result = await handle_admin_command(update)
            assert "недоступна" in result or "конфигурацию" in result

    @pytest.mark.asyncio
    async def test_handle_find_command(self):
        """Test handling /find command."""
        with patch("src.bot.tgbot._parse_find_command_args_helper", return_value=(0.5, "query")), \
             patch("src.bot.tgbot._send_find_results", new_callable=AsyncMock) as mock_send_results:
            
            mock_send_results.return_value = ""
            update = Mock()
            result = await handle_find_command("/find query", update)
            
            mock_send_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_find_command_parse_error(self):
        """Test handling /find command with parse error."""
        with patch("src.bot.tgbot._parse_find_command_args_helper", return_value=(None, "Error message")):
            update = Mock()
            result = await handle_find_command("/find", update)
            assert result == "Error message"

    @pytest.mark.asyncio
    async def test_handle_user_query(self, mock_bot):
        """Test handling user query."""
        result = await handle_user_query("test query", respond=True)
        assert result == "Response"
        mock_bot.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_user_query_error(self, mock_bot):
        """Test handling user query with error."""
        mock_bot.chat.side_effect = Exception("Error")
        result = await handle_user_query("test query", respond=True)
        assert "ошибка" in result or "Ошибка" in result

    @pytest.mark.asyncio
    async def test_route_command(self):
        """Test routing command."""
        with patch("src.bot.tgbot._get_command_and_args", return_value=("/start", "")), \
             patch("src.bot.tgbot._command_service", new_callable=Mock) as mock_service:
            
            # Since route_command uses handle_command_async from main_cli, we need to mock interactions
            # Easier to check if it tries to use command_service
            
            mock_service.dispatcher = Mock()
            
            # Mock handle_command_async to return something we expect
            with patch("src.app.main_cli.handle_command_async", new_callable=AsyncMock) as mock_handle:
                mock_handle.return_value = ("Start response", {})
                
                update = Mock()
                result = await route_command("/start", update)
                
                assert result == "Start response"


class TestUtilityFunctions:
    def test_register_command_group(self):
        """Test registering command group."""
        router = Mock()
        command_instance = Mock()
        command_instance.method1 = Mock()
        command_instance.method2 = Mock()
        
        methods = {
            "method1": "sub1",
            "method2": None
        }
        
        _register_command_group(router, "group", command_instance, methods)
        
        assert router.register.call_count == 2

    def test_get_profile_paths(self):
        """Test getting profile paths."""
        mock_pm = Mock()
        mock_pm.get_profile_paths.return_value = {"db_url": "sqlite:///test.db"}
        mock_ctx = Mock()
        mock_ctx.profile_manager = mock_pm
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            result = _get_profile_paths()
            assert result == {"db_url": "sqlite:///test.db"}

    def test_get_profile_paths_no_manager(self):
        """Test getting profile paths without manager."""
        mock_ctx = Mock()
        mock_ctx.profile_manager = None
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="profile_manager is not initialized"):
                _get_profile_paths()

    def test_create_admin_manager(self, tmp_path):
        """Test creating admin manager."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        
        with patch("src.bot.tgbot.AdminManager") as MockAdmin:
            mock_instance = Mock()
            MockAdmin.return_value = mock_instance
            
            result = _create_admin_manager(str(profile_dir))
            
            MockAdmin.assert_called_once_with(str(profile_dir))
            assert result == mock_instance

    def test_map_log_level_to_constant_string(self):
        """Test mapping log level string to constant."""
        from src.lib.syslog2 import LOG_INFO, LOG_DEBUG, LOG_WARNING
        
        assert _map_log_level_to_constant("INFO") == LOG_INFO
        assert _map_log_level_to_constant("DEBUG") == LOG_DEBUG
        assert _map_log_level_to_constant("WARNING") == LOG_WARNING

    def test_map_log_level_to_constant_int(self):
        """Test mapping log level int to constant."""
        from src.lib.syslog2 import LOG_INFO
        
        assert _map_log_level_to_constant(LOG_INFO) == LOG_INFO

    def test_get_bot_configuration(self):
        """Test getting bot configuration."""
        admin_manager = Mock()
        admin_manager.config.current_model = "gpt-4"
        
        args = SimpleNamespace(debug_rag=True, log_level="INFO", retrieval_type="hybrid")
        
        model, debug_rag, log_level, retrieval_type = _get_bot_configuration(admin_manager, args)
        
        assert model == "gpt-4"
        assert debug_rag is True
        assert retrieval_type == "hybrid"

    def test_create_legale_bot(self):
        """Test creating LegaleBot."""
        paths = {
            "db_url": "sqlite:///test.db",
            "vector_db_path": "/tmp/vec"
        }
        
        mock_pm = Mock()
        mock_pm.get_current_profile.return_value = "test_profile"
        mock_ctx = Mock()
        mock_ctx.profile_manager = mock_pm
        
        with patch("src.bot.tgbot.LegaleBot") as MockBot, \
             patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            
            mock_instance = Mock()
            MockBot.return_value = mock_instance
            
            result = _create_legale_bot(paths, "gpt-4", 4, False, "/tmp/profile", "hybrid")
            
            MockBot.assert_called_once()
            assert result == mock_instance

    def test_map_syslog2_to_logging_level(self):
        """Test mapping syslog2 to logging level."""
        import logging
        from src.lib.syslog2 import LOG_ERR, LOG_WARNING, LOG_INFO, LOG_DEBUG
        
        assert _map_syslog2_to_logging_level(LOG_ERR) == logging.ERROR
        assert _map_syslog2_to_logging_level(LOG_WARNING) == logging.WARNING
        assert _map_syslog2_to_logging_level(LOG_INFO) == logging.INFO
        assert _map_syslog2_to_logging_level(LOG_DEBUG) == logging.DEBUG

    def test_setup_logging(self):
        """Test setting up logging."""
        with patch("logging.StreamHandler") as MockHandler, \
             patch("logging.getLogger") as MockLogger:
            
            mock_handler = Mock()
            MockHandler.return_value = mock_handler
            mock_logger = Mock()
            MockLogger.return_value = mock_logger
            
            setup_logging(log_level="INFO", use_syslog=False)
            
            MockHandler.assert_called_once()
            mock_logger.addHandler.assert_called()

    def test_is_bot_mentioned(self):
        """Test checking if bot is mentioned."""
        message = Mock()
        message.text = "Hello @testbot"
        message.entities = [Mock(type="mention", offset=6, length=8)]
        
        result = is_bot_mentioned(message, "testbot", 123)
        assert result is True

    def test_is_bot_mentioned_no_entities(self):
        """Test checking mention without entities."""
        message = Mock()
        message.text = "Hello"
        message.entities = []
        
        result = is_bot_mentioned(message, "testbot", 123)
        assert result is False

    def test_extract_mention_text(self):
        """Test extracting mention text."""
        result = _extract_mention_text("@testbot hello", "testbot")
        assert result == "hello"

    def test_extract_mention_text_no_mention(self):
        """Test extracting mention text without mention."""
        result = _extract_mention_text("hello", "testbot")
        assert result is None

    def test_parse_search_command(self):
        """Test parsing search command."""
        result = _parse_search_command("поиск query")
        assert result == "query"

    def test_parse_search_command_find(self):
        """Test parsing search command with 'find'."""
        result = _parse_search_command("find query")
        assert result == "query"

    def test_parse_search_command_no_query(self):
        """Test parsing search command without query."""
        result = _parse_search_command("поиск")
        assert result is None

    def test_parse_search_mention(self):
        """Test parsing search mention."""
        message = Mock()
        message.text = "@testbot поиск query"
        message.entities = [Mock(type="mention", offset=0, length=9)]
        
        with patch("src.bot.tgbot.is_bot_mentioned", return_value=True), \
             patch("src.bot.tgbot._extract_mention_text", return_value="поиск query"), \
             patch("src.bot.tgbot._parse_search_command", return_value="query"):
            
            is_search, query = _parse_search_mention(message, "testbot", 123)
            assert is_search is True
            assert query == "query"

    @pytest.mark.asyncio
    async def test_send_message_parts_unified_empty(self):
        """Test sending empty message parts."""
        mock_app = Mock()
        mock_app.bot.send_message = AsyncMock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = mock_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            result = await _send_message_parts_unified(123, [], empty_message="No results")
            
            assert result == 0
            mock_app.bot.send_message.assert_called_once_with(chat_id=123, text="No results")

    @pytest.mark.asyncio
    async def test_send_message_parts_unified_with_parts(self):
        """Test sending message parts."""
        mock_app = Mock()
        mock_app.bot.send_message = AsyncMock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = mock_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            message_parts_list = [[{"content": "Part 1"}], [{"content": "Part 2"}]]
            result = await _send_message_parts_unified(123, message_parts_list)
            
            assert result == 2
            assert mock_app.bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_send_message_parts(self):
        """Test send_message_parts (deprecated wrapper)."""
        with patch("src.bot.tgbot._send_message_parts_unified", new_callable=AsyncMock) as mock_unified:
            mock_unified.return_value = 5
            
            result = await _send_message_parts(123, [])
            
            assert result == 5
            mock_unified.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_search_results(self):
        """Test sending search results."""
        with patch("src.bot.tgbot._send_message_parts_unified", new_callable=AsyncMock) as mock_unified:
            await _send_search_results(123, [], "query")
            
            mock_unified.assert_called_once()
            assert "query" in mock_unified.call_args[0][2] or "query" in str(mock_unified.call_args)

    @pytest.mark.asyncio
    async def test_should_ignore_message(self):
        """Test checking if message should be ignored."""
        result = await _should_ignore_message(False, 0, 123)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_ignore_message_not_ignored(self):
        """Test message that should not be ignored."""
        result = await _should_ignore_message(True, 1, 123)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_access(self):
        """Test checking access."""
        admin_manager = Mock()
        access_control = Mock()
        access_control.is_allowed.return_value = (True, None)
        mock_ctx = Mock()
        mock_ctx.admin_manager = admin_manager
        mock_ctx.access_control = access_control
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            is_allowed, reason = _check_access(123, 456, True, False, None)
            
            assert is_allowed is True
            assert reason is None

    @pytest.mark.asyncio
    async def test_check_access_denied(self):
        """Test access denied."""
        admin_manager = Mock()
        access_control = Mock()
        access_control.is_allowed.return_value = (False, "Not allowed")
        mock_ctx = Mock()
        mock_ctx.admin_manager = admin_manager
        mock_ctx.access_control = access_control
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            is_allowed, reason = _check_access(123, 456, True, False, None)
            
            assert is_allowed is False
            assert reason == "Not allowed"

    @pytest.mark.asyncio
    async def test_handle_public_commands_id(self):
        """Test handling /id command."""
        message = Mock()
        message.from_user.id = 123
        mock_app = Mock()
        mock_app.bot.send_message = AsyncMock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = mock_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            result = await _handle_public_commands(message, "/id", 456)
            
            assert result is True
            mock_app.bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_public_commands_help(self):
        """Test handling /help command."""
        message = Mock()
        
        with patch("src.bot.tgbot.handle_help_command", new_callable=AsyncMock) as mock_help, \
             patch("src.bot.tgbot._send_handler_response", new_callable=AsyncMock) as mock_send:
            
            mock_help.return_value = "Help text"
            
            result = await _handle_public_commands(message, "/help", 456)
            
            assert result is True
            mock_help.assert_called_once()
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_public_commands_admin_set(self):
        """Test handling /admin_set command."""
        message = Mock()
        
        with patch("src.bot.tgbot.handle_admin_set_command", new_callable=AsyncMock) as mock_set, \
             patch("src.bot.tgbot._send_handler_response", new_callable=AsyncMock) as mock_send:
            
            mock_set.return_value = "Success"
            
            result = await _handle_public_commands(message, "/admin_set password", 456)
            
            assert result is True
            mock_set.assert_called_once()
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_public_commands_not_public(self):
        """Test handling non-public command."""
        message = Mock()
        
        result = await _handle_public_commands(message, "/unknown", 456)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_determine_response_decision(self):
        """Test determining response decision."""
        message = Mock()
        message.chat.type = "private"
        
        admin_manager = Mock()
        admin_manager.config.response_frequency = 1
        
        frequency_controller = Mock()
        frequency_controller.should_respond.return_value = (True, "mentioned")
        
        mock_app = Mock()
        mock_app.bot.username = "testbot"
        mock_app.bot.id = 123
        
        mock_ctx = Mock()
        mock_ctx.admin_manager = admin_manager
        mock_ctx.telegram_app = mock_app
        mock_ctx.frequency_controller = frequency_controller
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            respond, reason = await _determine_response_decision(message, False, True, 456)
            
            assert respond is True
            assert reason == "mentioned"

    @pytest.mark.asyncio
    async def test_send_response_if_available(self):
        """Test sending response if available."""
        mock_app = Mock()
        mock_app.bot.send_message = AsyncMock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = mock_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            await _send_response_if_available("Response text", 123, True, True)
            
            mock_app.bot.send_message.assert_called_once_with(chat_id=123, text="Response text")

    @pytest.mark.asyncio
    async def test_send_response_if_available_none(self):
        """Test sending response when response is None."""
        mock_app = Mock()
        mock_app.bot.send_message = AsyncMock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = mock_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            await _send_response_if_available(None, 123, True, True)
            
            mock_app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_command_message(self):
        """Test processing command message."""
        with patch("src.bot.tgbot.route_command", new_callable=AsyncMock) as mock_route, \
             patch("src.bot.tgbot.handle_user_query", new_callable=AsyncMock) as mock_query:
            
            mock_route.return_value = None
            mock_query.return_value = "Response"
            
            update = Mock()
            
            result = await _process_command_message("/unknown", update, True)
            
            assert result == "Response"
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_regular_message(self):
        """Test processing regular message."""
        with patch("src.bot.tgbot.handle_user_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"
            
            config = Mock()
            config.response_frequency = 1
            
            with patch("src.bot.tgbot._should_ignore_message", new_callable=AsyncMock) as mock_ignore:
                mock_ignore.return_value = False
                
                result = await _process_regular_message("text", True, config, 123)
                
                assert result == "Response"
                mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_regular_message_ignored(self):
        """Test processing regular message that should be ignored."""
        with patch("src.bot.tgbot.handle_user_query", new_callable=AsyncMock) as mock_query:
            config = Mock()
            config.response_frequency = 0
            
            with patch("src.bot.tgbot._should_ignore_message", new_callable=AsyncMock) as mock_ignore:
                mock_ignore.return_value = True
                
                result = await _process_regular_message("text", False, config, 123)
                
                assert result is None
                mock_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_search_mention(self):
        """Test handling search mention."""
        message = Mock()
        bot_instance = Mock()
        bot_instance.retrieval = Mock()
        bot_instance.db = Mock()
        mock_ctx = Mock()
        mock_ctx.bot_instance = bot_instance
        
        with patch("src.bot.tgbot._parse_search_mention", return_value=(True, "query")), \
             patch("src.bot.tgbot.search_message_contents", return_value=[]), \
             patch("src.bot.tgbot._send_search_results", new_callable=AsyncMock), \
             patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx):
            
            result = await _handle_search_mention(message, "testbot", 123, 456)
            
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_search_mention_no_search(self):
        """Test handling message without search mention."""
        message = Mock()
        
        with patch("src.bot.tgbot._parse_search_mention", return_value=(False, "")):
            result = await _handle_search_mention(message, "testbot", 123, 456)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_parse_webhook_update(self):
        """Test parsing webhook update."""
        request = Mock()
        request.json = AsyncMock(return_value={"update_id": 1})
        
        telegram_app = Mock()
        telegram_app.bot = Mock()
        mock_ctx = Mock()
        mock_ctx.telegram_app = telegram_app
        
        with patch("src.bot.tgbot.get_runtime_context", return_value=mock_ctx), \
             patch("src.bot.tgbot.Update.de_json") as mock_de_json:
            
            mock_update = Mock()
            mock_de_json.return_value = mock_update
            
            result = await _parse_webhook_update(request, mock_ctx)
            
            assert result == mock_update

    @pytest.mark.asyncio
    async def test_parse_webhook_update_error(self):
        """Test parsing webhook update with error."""
        request = Mock()
        request.json = AsyncMock(side_effect=Exception("Parse error"))
        mock_ctx = Mock()
        
        result = await _parse_webhook_update(request, mock_ctx)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_process_webhook_update_document(self):
        """Test processing document update."""
        update = Mock()
        update.message = Mock()
        update.message.document = Mock()
        
        ingest_commands = Mock()
        ingest_commands.handle_file_upload = AsyncMock(return_value="File uploaded")
        admin_manager = Mock()
        mock_ctx = Mock()
        mock_ctx.ingest_commands = ingest_commands
        mock_ctx.admin_manager = admin_manager
        
        with patch("src.bot.tgbot.process_document_update", new_callable=AsyncMock) as mock_process_doc:
            mock_process_doc.return_value = "File uploaded"
            
            result = await _process_webhook_update(update, mock_ctx)
            
            assert result == "File uploaded"

    @pytest.mark.asyncio
    async def test_process_webhook_update_text(self):
        """Test processing text update."""
        update = Mock()
        update.message = Mock()
        update.message.text = "Hello"
        update.message.document = None
        
        with patch("src.bot.tgbot.process_text_update", new_callable=AsyncMock):
            result = await _process_webhook_update(update)
            
            assert result is None

    def test_create_app(self):
        """Test creating FastAPI app."""
        with patch("src.bot.tgbot.FastAPI") as MockFastAPI, \
             patch("src.bot.tgbot.lifespan") as mock_lifespan:
            
            mock_app = Mock()
            MockFastAPI.return_value = mock_app
            
            app = create_app()
            
            MockFastAPI.assert_called_once()
            assert app == mock_app

    @pytest.mark.asyncio
    async def test_handle_message(self):
        """Test handling message."""
        update = Mock()
        update.message = Mock()
        update.message.text = "/start"
        update.message.chat_id = 123
        update.message.from_user = Mock()
        update.message.from_user.id = 456
        update.message.chat.type = "private"
        
        with patch("src.bot.tgbot._handle_public_commands_step", new_callable=AsyncMock) as mock_public:
            mock_public.return_value = True
            
            await handle_message(update)
            
            mock_public.assert_called_once()
