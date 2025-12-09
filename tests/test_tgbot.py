"""
Integration tests for Telegram Bot Webhook.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Fix path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import app
from src.bot.tgbot import app
import src.bot.tgbot as tgbot_module

@pytest.fixture
def mock_deps():
    # Create mocks
    bot_mock = MagicMock()
    bot_mock.chat.return_value = "Mocked Response"
    
    admin_mock = MagicMock()
    admin_mock.config = MagicMock()
    admin_mock.config.allowed_chats = []
    admin_mock.config.response_frequency = 1
    admin_mock.verify_password.return_value = False
    admin_mock.is_admin.return_value = False
    
    tg_mock = MagicMock()
    tg_mock.bot.send_message = AsyncMock()
    tg_mock.bot.username = "testbot"
    tg_mock.bot.id = 12345
    
    # Mock access_control service
    access_control_mock = MagicMock()
    access_control_mock.is_allowed.return_value = (True, None)  # Allow by default
    
    # Custom side effect for Update.de_json to return dynamic mock
    def mock_de_json(data, bot):
        m = MagicMock()
        m.update_id = data.get('update_id')
        if 'message' in data:
            msg_data = data['message']
            m.message.message_id = msg_data.get('message_id')
            m.message.chat_id = msg_data['chat']['id']  # Note: helper property usually
            m.message.chat.id = msg_data['chat']['id']
            m.message.chat.type = msg_data['chat'].get('type', 'private')
            m.message.from_user.id = msg_data['from']['id']
            m.message.text = msg_data.get('text')
            m.message.entities = []  # Add entities for mention checking
            
            # Allow accessing unknown attributes as None/Mock
        return m

    # Apply patches
    with patch.object(tgbot_module, 'bot_instance', bot_mock), \
         patch.object(tgbot_module, 'admin_manager', admin_mock), \
         patch.object(tgbot_module, 'telegram_app', tg_mock), \
         patch.object(tgbot_module, 'access_control', access_control_mock), \
         patch('src.bot.tgbot.Update.de_json', side_effect=mock_de_json):
        
        yield {
            'bot': bot_mock,
            'admin': admin_mock,
            'tg': tg_mock,
            'access_control': access_control_mock
        }

@pytest.fixture
def client(mock_deps):
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_webhook_message_allowed(client, mock_deps):
    """Test standard message processing for allowed chat."""
    # Allow the chat
    mock_deps['admin'].config.allowed_chats = [123]
    
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "date": 123456,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Test"},
            "text": "Hello"
        }
    }
    
    response = client.post("/webhook", json=update)
    assert response.status_code == 200
    
    # Should call chat (may include system_prompt_template)
    call_args = mock_deps['bot'].chat.call_args
    assert call_args is not None
    assert call_args[0][0] == "Hello"
    assert call_args[1].get('respond') == True
    mock_deps['tg'].bot.send_message.assert_called()

def test_webhook_whitelist_block(client, mock_deps):
    """Test blocking when chat is not in whitelist."""
    # Setup whitelist
    mock_deps['admin'].config.allowed_chats = [999]
    
    # Configure access_control to block this chat
    mock_deps['access_control'].is_allowed.return_value = (False, "not in whitelist")
    
    update = {
        "update_id": 2,
        "message": {
            "message_id": 2,
            "date": 123456,
            "chat": {"id": 123, "type": "private"}, # Blocked
            "from": {"id": 1, "is_bot": False, "first_name": "Test"},
            "text": "Hello"
        }
    }
    
    client.post("/webhook", json=update)
    
    # Should NOT call chat
    mock_deps['bot'].chat.assert_not_called()


def test_webhook_private_admin(client, mock_deps):
    """Test admin access in private chat (overrides whitelist)."""
    mock_deps['admin'].config.allowed_chats = []
    mock_deps['admin'].is_admin.return_value = True
    
    update = {
        "update_id": 10,
        "message": {
            "message_id": 1,
            "date": 123,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 1},
            "text": "Hello"
        }
    }
    client.post("/webhook", json=update)
    mock_deps['bot'].chat.assert_called()


@pytest.mark.asyncio
async def test_handle_find_command_success(mock_deps):
    """Test successful /find command execution."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from src.bot.tgbot import MessageHandler
    
    # Setup mocks
    mock_bot = MagicMock()
    mock_bot.db = MagicMock()
    mock_bot.retrieval = MagicMock()
    
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    
    # Mock search_message_contents to return results
    mock_message_parts = [
        [{"content": "Result 1", "date": "2025-01-01", "id": 1, "sender": "User1"}],
        [{"content": "Result 2", "date": "2025-01-02", "id": 2, "sender": "User2"}]
    ]
    
    handler = MessageHandler(mock_bot)
    
    with patch('src.bot.tgbot.telegram_app') as mock_telegram_app:
        mock_telegram_app.bot.send_message = AsyncMock()
        with patch('src.core.message_search.search_message_contents', return_value=mock_message_parts):
            result = await handler.handle_find_command("/find test query", mock_update)
    
    # Should return None (messages sent directly)
    assert result is None
    # Verify messages were sent
    assert mock_telegram_app.bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_handle_find_command_empty_query(mock_deps):
    """Test /find command with empty query."""
    from unittest.mock import MagicMock
    from src.bot.tgbot import MessageHandler
    
    mock_bot = MagicMock()
    handler = MessageHandler(mock_bot)
    
    mock_update = MagicMock()
    
    result = await handler.handle_find_command("/find", mock_update)
    
    assert result is not None
    assert "Использование" in result
    assert "/find <запрос>" in result


@pytest.mark.asyncio
async def test_handle_find_command_no_results(mock_deps):
    """Test /find command when no results found."""
    from unittest.mock import MagicMock, patch
    from src.bot.tgbot import MessageHandler
    
    mock_bot = MagicMock()
    mock_bot.db = MagicMock()
    mock_bot.retrieval = MagicMock()
    
    handler = MessageHandler(mock_bot)
    
    mock_update = MagicMock()
    
    with patch('src.core.message_search.search_message_contents', return_value=[]):
        result = await handler.handle_find_command("/find nonexistent query", mock_update)
    
    assert result is not None
    assert "ничего не найдено" in result


@pytest.mark.asyncio
async def test_handle_find_command_send_error(mock_deps):
    """Test /find command when sending messages fails."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from src.bot.tgbot import MessageHandler
    
    mock_bot = MagicMock()
    mock_bot.db = MagicMock()
    mock_bot.retrieval = MagicMock()
    
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    
    mock_message_parts = [
        [{"content": "Result 1", "date": "2025-01-01", "id": 1, "sender": "User1"}]
    ]
    
    handler = MessageHandler(mock_bot)
    
    with patch('src.bot.tgbot.telegram_app') as mock_telegram_app:
        mock_telegram_app.bot.send_message = AsyncMock(side_effect=Exception("Send error"))
        with patch('src.core.message_search.search_message_contents', return_value=mock_message_parts):
            result = await handler.handle_find_command("/find test query", mock_update)
    
    assert result is not None
    assert "Ошибка при отправке" in result


@pytest.mark.asyncio
async def test_handle_find_command_search_error(mock_deps):
    """Test /find command when search fails."""
    from unittest.mock import MagicMock, patch
    from src.bot.tgbot import MessageHandler
    
    mock_bot = MagicMock()
    mock_bot.db = MagicMock()
    mock_bot.retrieval = MagicMock()
    
    handler = MessageHandler(mock_bot)
    
    mock_update = MagicMock()
    
    with patch('src.core.message_search.search_message_contents', side_effect=Exception("Search error")):
        result = await handler.handle_find_command("/find test query", mock_update)
    
    assert result is not None
    assert "Ошибка при выполнении поиска" in result


@pytest.mark.asyncio
async def test_lifespan_startup_shutdown(mock_deps):
    """Test lifespan context manager startup and shutdown."""
    from unittest.mock import MagicMock, patch, AsyncMock
    from contextlib import asynccontextmanager
    import src.bot.tgbot as tgbot_module
    
    # Mock ProfileManager
    mock_profile_manager = MagicMock()
    mock_profile_manager.get_current_profile.return_value = "test_profile"
    
    # Mock init_runtime_for_current_profile
    mock_init_runtime = AsyncMock()
    
    # Mock Telegram application
    mock_telegram_app = MagicMock()
    mock_telegram_app.bot = MagicMock()
    mock_telegram_app.application = MagicMock()
    mock_telegram_app.application.initialize = AsyncMock()
    mock_telegram_app.application.start = AsyncMock()
    mock_telegram_app.application.stop = AsyncMock()
    mock_telegram_app.application.shutdown = AsyncMock()
    
    with patch('src.bot.tgbot.ProfileManager', return_value=mock_profile_manager), \
         patch('src.bot.tgbot.init_runtime_for_current_profile', mock_init_runtime), \
         patch('src.bot.tgbot.telegram_app', mock_telegram_app), \
         patch('src.bot.tgbot.os.getenv', return_value="test_token"), \
         patch('src.bot.tgbot.Application') as mock_app_class:
        
        # Import lifespan function
        from src.bot.tgbot import lifespan
        
        # Test as async context manager
        async with lifespan(tgbot_module.app):
            # During startup
            mock_init_runtime.assert_called_once()
            mock_telegram_app.application.initialize.assert_called_once()
            mock_telegram_app.application.start.assert_called_once()
        
        # During shutdown
        mock_telegram_app.application.stop.assert_called_once()
        mock_telegram_app.application.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_profile_manager_error(mock_deps):
    """Test lifespan when ProfileManager initialization fails."""
    from unittest.mock import MagicMock, patch
    import src.bot.tgbot as tgbot_module
    
    with patch('src.bot.tgbot.ProfileManager', side_effect=Exception("Profile manager error")):
        from src.bot.tgbot import lifespan
        
        with pytest.raises(RuntimeError, match="Profile manager initialization failed"):
            async with lifespan(tgbot_module.app):
                pass


@pytest.mark.asyncio
async def test_lifespan_runtime_init_error(mock_deps):
    """Test lifespan when runtime initialization fails."""
    from unittest.mock import MagicMock, patch, AsyncMock
    import src.bot.tgbot as tgbot_module
    
    mock_profile_manager = MagicMock()
    mock_profile_manager.get_current_profile.return_value = "test_profile"
    
    mock_init_runtime = AsyncMock(side_effect=Exception("Runtime init error"))
    
    with patch('src.bot.tgbot.ProfileManager', return_value=mock_profile_manager), \
         patch('src.bot.tgbot.init_runtime_for_current_profile', mock_init_runtime):
        
        from src.bot.tgbot import lifespan
        
        with pytest.raises(Exception, match="Runtime init error"):
            async with lifespan(tgbot_module.app):
                pass

def test_webhook_private_command(client, mock_deps):
    """Test command allowed in private chat even if not admin/whitelisted."""
    mock_deps['admin'].config.allowed_chats = []
    mock_deps['admin'].is_admin.return_value = False
    
    update = {
        "update_id": 11,
        "message": {
            "message_id": 1,
            "date": 123,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 2},
            "text": "/start"
        }
    }
    client.post("/webhook", json=update)
    
    # Check that send_message was called (response to /start)
    mock_deps['tg'].bot.send_message.assert_called()
    # Check that bot.chat was NOT called (it's a command)
    mock_deps['bot'].chat.assert_not_called()
