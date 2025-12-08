"""
Extended tests for Telegram Bot MessageHandler and utilities.

Tests cover:
- MessageHandler command routing
- Individual command handlers
- is_bot_mentioned() function
- Access control integration
- Frequency control integration
"""

import pytest
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from telegram import Update, Message, Chat, User
from src.bot.tgbot import MessageHandler, is_bot_mentioned


class TestMessageHandler:
    """Tests for MessageHandler class."""
    
    @pytest.fixture
    def setup_handler(self):
        """Setup MessageHandler with mocked dependencies."""
        bot_mock = Mock()
        bot_mock.chat.return_value = "Bot response"
        bot_mock.reset_context.return_value = "Context reset"
        bot_mock.get_token_usage.return_value = {
            'current_tokens': 1000,  # Fixed: was 'total_tokens'
            'max_tokens': 14000,
            'percentage': 7.14
        }
        bot_mock.get_model.return_value = None  # Fixed: method returns None
        bot_mock.get_current_model.return_value = ("gpt-3.5-turbo", 1, 3)
        
        admin_mock = Mock()
        admin_mock.is_admin.return_value = False
        admin_mock.verify_password.return_value = False
        admin_mock.set_admin.return_value = None
        
        router_mock = Mock()
        router_mock.route.return_value = "Admin command response"
        
        handler = MessageHandler(bot_mock, admin_mock, router_mock)
        
        return {
            'handler': handler,
            'bot': bot_mock,
            'admin': admin_mock,
            'router': router_mock
        }
    
    @pytest.mark.asyncio
    async def test_handle_start_command(self, setup_handler):
        """Test /start command returns welcome message."""
        handler = setup_handler['handler']
        result = await handler.handle_start_command()
        
        assert "Привет" in result or "юридический" in result
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_handle_help_command(self, setup_handler):
        """Test /help command returns help text."""
        handler = setup_handler['handler']
        result = await handler.handle_help_command()
        
        assert "/start" in result
        assert "/help" in result
        assert "/reset" in result
        assert "/tokens" in result
        assert "/model" in result
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_handle_reset_command(self, setup_handler):
        """Test /reset command calls bot.reset_context()."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        
        result = await handler.handle_reset_command()
        
        bot.reset_context.assert_called_once()
        assert result == "Context reset"
    
    @pytest.mark.asyncio
    async def test_handle_tokens_command_low_usage(self, setup_handler):
        """Test /tokens command with low token usage."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        bot.get_token_usage.return_value = {
            'current_tokens': 1000,  # Fixed: was 'total_tokens'
            'max_tokens': 14000,
            'percentage': 7.14
        }
        
        result = await handler.handle_tokens_command()
        
        bot.get_token_usage.assert_called_once()
        assert "1000" in result or "1,000" in result
        assert "14000" in result or "14,000" in result
        assert "7" in result  # percentage
    
    @pytest.mark.asyncio
    async def test_handle_tokens_command_medium_usage(self, setup_handler):
        """Test /tokens command with medium token usage (50-80%)."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        bot.get_token_usage.return_value = {
            'current_tokens': 9000,  # Fixed: was 'total_tokens'
            'max_tokens': 14000,
            'percentage': 64.3
        }
        
        result = await handler.handle_tokens_command()
        
        assert "9000" in result or "9,000" in result
        assert "64" in result
        assert "" in result  # info emoji for medium usage
    
    @pytest.mark.asyncio
    async def test_handle_tokens_command_high_usage(self, setup_handler):
        """Test /tokens command with high token usage (>80%)."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        bot.get_token_usage.return_value = {
            'current_tokens': 12000,  # Fixed: was 'total_tokens'
            'max_tokens': 14000,
            'percentage': 85.7
        }
        
        result = await handler.handle_tokens_command()
        
        assert "12000" in result or "12,000" in result
        assert "85" in result
        assert "" in result  # warning emoji for high usage
    
    @pytest.mark.asyncio
    async def test_handle_model_command(self, setup_handler):
        """Test /model command getes model."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        bot.get_model.return_value = "Модель переключена на: gpt-4\n(2/3)"
        
        result = await handler.handle_model_command()
        
        bot.get_model.assert_called_once()
        assert "Модель переключена" in result or "gpt-4" in result
    
    @pytest.mark.asyncio
    async def test_handle_admin_set_command_no_password(self, setup_handler):
        """Test /admin_set without password."""
        handler = setup_handler['handler']
        message_mock = Mock()
        message_mock.from_user.id = 12345
        
        result = await handler.handle_admin_set_command("/admin_set", message_mock)
        
        assert "пароль" in result.lower()
    
    @pytest.mark.asyncio
    async def test_handle_admin_set_command_wrong_password(self, setup_handler):
        """Test /admin_set with wrong password."""
        handler = setup_handler['handler']
        admin = setup_handler['admin']
        admin.verify_password.return_value = False
        
        message_mock = Mock()
        message_mock.from_user.id = 12345
        
        result = await handler.handle_admin_set_command("/admin_set wrong_password", message_mock)
        
        admin.verify_password.assert_called_once_with("wrong_password")
        assert "Неверный пароль" in result
    
    @pytest.mark.asyncio
    async def test_handle_admin_set_command_correct_password(self, setup_handler):
        """Test /admin_set with correct password."""
        handler = setup_handler['handler']
        admin = setup_handler['admin']
        admin.verify_password.return_value = True
        
        message_mock = Mock()
        message_mock.from_user.id = 12345
        message_mock.from_user.username = "testuser"
        message_mock.from_user.first_name = "Test"
        message_mock.from_user.last_name = "User"
        
        result = await handler.handle_admin_set_command("/admin_set correct_password", message_mock)
        
        admin.verify_password.assert_called_once_with("correct_password")
        admin.set_admin.assert_called_once_with(12345, "testuser", "Test", "User")
        assert "назначены администратором" in result
    
    @pytest.mark.asyncio
    async def test_handle_admin_get_command_not_admin(self, setup_handler):
        """Test /admin_get for non-admin user."""
        handler = setup_handler['handler']
        admin = setup_handler['admin']
        admin.is_admin.return_value = False
        
        result = await handler.handle_admin_get_command(12345)
        
        assert "администратору" in result.lower()
    
    @pytest.mark.asyncio
    async def test_handle_admin_get_command_is_admin(self, setup_handler):
        """Test /admin_get for admin user."""
        handler = setup_handler['handler']
        admin = setup_handler['admin']
        admin.is_admin.return_value = True
        admin.get_admin.return_value = {
            'user_id': 12345,
            'username': 'admin_user',
            'full_name': 'Admin User'
        }
        
        result = await handler.handle_admin_get_command(12345)
        
        assert "Администратор бота" in result
        assert "12345" in result
    
    @pytest.mark.asyncio
    async def test_handle_user_query_respond_true(self, setup_handler):
        """Test user query with respond=True."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        admin = setup_handler['admin']
        
        result = await handler.handle_user_query("What is Python?", respond=True)
        
        bot.chat.assert_called_once_with("What is Python?", respond=True, system_prompt_template=admin.config.system_prompt)
        assert result == "Bot response"
    
    @pytest.mark.asyncio
    async def test_handle_user_query_respond_false(self, setup_handler):
        """Test user query with respond=False (silent processing)."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        admin = setup_handler['admin']
        # Mock silent response (usually None)
        bot.chat.return_value = None
        
        result = await handler.handle_user_query("What is Python?", respond=False)
        
        bot.chat.assert_called_with("What is Python?", respond=False, system_prompt_template=admin.config.system_prompt)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_route_command_start(self, setup_handler):
        """Test routing /start command."""
        handler = setup_handler['handler']
        update_mock = Mock()
        
        result = await handler.route_command("/start", update_mock)
        
        assert "Привет" in result or "юридический" in result
    
    @pytest.mark.asyncio
    async def test_route_command_help(self, setup_handler):
        """Test routing /help command."""
        handler = setup_handler['handler']
        update_mock = Mock()
        
        result = await handler.route_command("/help", update_mock)
        
        assert "/start" in result
    
    @pytest.mark.asyncio
    async def test_route_command_reset(self, setup_handler):
        """Test routing /reset command."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        update_mock = Mock()
        
        result = await handler.route_command("/reset", update_mock)
        
        bot.reset_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_command_tokens(self, setup_handler):
        """Test routing /tokens command."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        update_mock = Mock()
        
        result = await handler.route_command("/tokens", update_mock)
        
        bot.get_token_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_command_model(self, setup_handler):
        """Test routing /model command."""
        handler = setup_handler['handler']
        bot = setup_handler['bot']
        update_mock = Mock()
        
        result = await handler.route_command("/model", update_mock)
        
        bot.get_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_command_admin_set(self, setup_handler):
        """Test routing /admin_set command."""
        handler = setup_handler['handler']
        update_mock = Mock()
        update_mock.message.from_user.id = 12345
        
        result = await handler.route_command("/admin_set password123", update_mock)
        
        assert "пароль" in result.lower() or "администратор" in result.lower()
    
    @pytest.mark.asyncio
    async def test_route_command_admin_get(self, setup_handler):
        """Test routing /admin_get command."""
        handler = setup_handler['handler']
        update_mock = Mock()
        update_mock.message.from_user.id = 12345
        
        result = await handler.route_command("/admin_get", update_mock)
        
        assert "администратор" in result.lower()
    
    @pytest.mark.asyncio
    async def test_route_command_admin(self, setup_handler):
        """Test routing /admin command."""
        handler = setup_handler['handler']
        router = setup_handler['router']
        update_mock = Mock()
        
        result = await handler.route_command("/admin", update_mock)
        
        router.route.assert_called_once()
    
    # test_route_command_id removed as /id is handled in handle_message, not route_command
    
    @pytest.mark.asyncio
    async def test_route_command_unknown(self, setup_handler):
        """Test routing unknown command."""
        handler = setup_handler['handler']
        update_mock = Mock()
        
        result = await handler.route_command("/unknown_command", update_mock)
        
        # Should return None for unknown command (handled as text later)
        assert result is None


class TestIsBotMentioned:
    """Tests for is_bot_mentioned() function."""
    
    def test_no_mention_no_entities(self):
        """Test message without mention or entities."""
        message = Mock()
        message.text = "Hello, how are you?"
        message.entities = None
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is False
    
    def test_mention_by_username_with_entity(self):
        """Test message with @username mention entity."""
        message = Mock()
        message.text = "Hello @testbot, how are you?"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 6
        entity.length = 8  # @testbot
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_mention_by_username_case_insensitive(self):
        """Test @username mention is case-insensitive."""
        message = Mock()
        message.text = "Hello @TestBot, how are you?"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 6
        entity.length = 8  # @TestBot
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_mention_entity_text_mention(self):
        """Test message with text_mention entity."""
        message = Mock()
        message.text = "Hello bot, how are you?"
        
        entity = Mock()
        entity.type = "text_mention"
        entity.user = Mock()
        entity.user.id = 12345
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_mention_entity_mention_type(self):
        """Test message with mention entity type."""
        message = Mock()
        message.text = "Hello @testbot"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 6
        entity.length = 8
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_mention_entity_wrong_user(self):
        """Test text_mention for different user."""
        message = Mock()
        message.text = "Hello other user"
        
        entity = Mock()
        entity.type = "text_mention"
        entity.user = Mock()
        entity.user.id = 99999  # Different bot ID
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is False
    
    def test_mention_multiple_entities(self):
        """Test message with multiple entities, one is bot mention."""
        message = Mock()
        message.text = "Hello @user and @testbot"
        
        entity1 = Mock()
        entity1.type = "mention"
        entity1.offset = 6
        entity1.length = 5
        
        entity2 = Mock()
        entity2.type = "mention"
        entity2.offset = 16
        entity2.length = 8
        
        message.entities = [entity1, entity2]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_no_mention_empty_entities(self):
        """Test message with empty entities list."""
        message = Mock()
        message.text = "Hello, how are you?"
        message.entities = []
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is False
    
    def test_mention_in_middle_of_text_with_entity(self):
        """Test @username mention in middle of message with entity."""
        message = Mock()
        message.text = "I think @testbot can help with this question"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 8
        entity.length = 8  # @testbot
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is True
    
    def test_no_mention_similar_username(self):
        """Test similar but different username."""
        message = Mock()
        message.text = "Hello @testbot2, how are you?"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 6
        entity.length = 9  # @testbot2
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is False
    
    def test_mention_with_underscore(self):
        """Test bot username with underscore."""
        message = Mock()
        message.text = "Hello @test_bot, how are you?"
        
        entity = Mock()
        entity.type = "mention"
        entity.offset = 6
        entity.length = 9  # @test_bot
        
        message.entities = [entity]
        
        result = is_bot_mentioned(message, "test_bot", 12345)
        
        assert result is True
    
    def test_no_entities_returns_false(self):
        """Test that without entities, function returns False even with @mention in text."""
        message = Mock()
        message.text = "Hello @testbot"
        message.entities = None
        
        result = is_bot_mentioned(message, "testbot", 12345)
        
        assert result is False


class TestHandleMessage:
    """Tests for handle_message() function."""
    
    @pytest.fixture
    def mock_deps(self):
        """Mock all global dependencies for handle_message."""
        with patch("src.bot.tgbot.telegram_app") as app_mock, \
             patch("src.bot.tgbot.admin_manager") as admin_mock, \
             patch("src.bot.tgbot.access_control") as access_mock, \
             patch("src.bot.tgbot.frequency_controller") as freq_mock, \
             patch("src.bot.tgbot.bot_instance") as bot_mock, \
             patch("src.bot.tgbot.admin_router") as router_mock:
            
            # Setup default behavior
            app_mock.bot.id = 12345
            app_mock.bot.username = "testbot"
            app_mock.bot.send_message = AsyncMock()
            
            admin_mock.config.response_frequency = 1
            
            access_mock.is_allowed.return_value = (True, None)
            
            freq_mock.should_respond.return_value = (True, None)
            
            bot_mock.chat.return_value = "Bot response"
            
            yield {
                'app': app_mock,
                'admin': admin_mock,
                'access': access_mock,
                'freq': freq_mock,
                'bot': bot_mock,
                'router': router_mock
            }

    @pytest.mark.asyncio
    async def test_handle_id_command(self, mock_deps):
        """Test /id command execution."""
        from src.bot.tgbot import handle_message
        
        update = Mock()
        update.message.text = "/id"
        update.message.chat_id = 999
        update.message.from_user.id = 111
        update.message.chat.type = "private"
        
        await handle_message(update)
        
        mock_deps['app'].bot.send_message.assert_called_once()
        call_args = mock_deps['app'].bot.send_message.call_args[1]
        assert call_args['chat_id'] == 999
        assert "Chat ID: `999`" in call_args['text']

    @pytest.mark.asyncio
    async def test_missing_admin_manager(self, mock_deps):
        """Test behavior when admin_manager is None."""
        from src.bot.tgbot import handle_message
        
        # Unmock admin_manager to make it None (or set patch to None if possible, 
        # but here we can just set the global if we were not mocking it directly.
        # Since we mocked it, we need to make the mock context manager return None for it?
        # Easier way: simulate None by starting a new patch context where it is None
        with patch("src.bot.tgbot.admin_manager", None):
            update = Mock()
            update.message.text = "hello"
            update.message.chat_id = 999
            
            await handle_message(update)
            
            # Should simply return without error and without sending message
            mock_deps['app'].bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_access_denied(self, mock_deps):
        """Test behavior when access is denied."""
        from src.bot.tgbot import handle_message
        
        mock_deps['access'].is_allowed.return_value = (False, "Denied")
        
        update = Mock()
        update.message.text = "hello"
        update.message.chat_id = 999
        update.message.chat.type = "private"
        update.message.from_user.id = 111
        
        await handle_message(update)
        
        mock_deps['app'].bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_frequency_limited_freq_zero(self, mock_deps):
        """Test behavior when freq=0 (should skip completely)."""
        from src.bot.tgbot import handle_message
        
        # Setup: Freq=0, No mention
        mock_deps['admin'].config.response_frequency = 0
        mock_deps['freq'].should_respond.return_value = (False, "freq_zero_no_mention")
        
        update = Mock()
        update.message.text = "hello"
        update.message.chat_id = 999
        update.message.chat.type = "group"
        update.message.from_user.id = 111
        update.message.entities = []
        
        await handle_message(update)
        
        # Should NOT call chat (skips history)
        mock_deps['bot'].chat.assert_not_called()
        mock_deps['app'].bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_frequency_limited_freq_n(self, mock_deps):
        """Test behavior when freq>1 (should record but silence)."""
        from src.bot.tgbot import handle_message
        
        # Setup: Freq=5, Skip message
        mock_deps['admin'].config.response_frequency = 5
        mock_deps['freq'].should_respond.return_value = (False, "freq_skip_1")
        mock_deps['bot'].chat.return_value = None  # Silent response
        
        update = Mock()
        update.message.text = "hello"
        update.message.chat_id = 999
        update.message.chat.type = "group"
        update.message.from_user.id = 111
        update.message.entities = []
        
        await handle_message(update)
        
        # Should call chat with respond=False (records history)
        mock_deps['bot'].chat.assert_called_with("hello", respond=False, system_prompt_template=mock_deps['admin'].config.system_prompt)
        mock_deps['app'].bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_message(self, mock_deps):
        """Test successful text message processing."""
        from src.bot.tgbot import handle_message
        
        update = Mock()
        update.message.text = "hello"
        update.message.chat_id = 999
        update.message.chat.type = "private"
        update.message.from_user.id = 111
        update.message.entities = []
        
        await handle_message(update)
        
        mock_deps['bot'].chat.assert_called_with("hello", respond=True, system_prompt_template=mock_deps['admin'].config.system_prompt)
        mock_deps['app'].bot.send_message.assert_called_once_with(
            chat_id=999,
            text="Bot response"
        )
