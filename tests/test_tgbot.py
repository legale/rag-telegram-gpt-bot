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
