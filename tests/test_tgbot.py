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
    
    tg_mock = MagicMock()
    tg_mock.bot.send_message = AsyncMock()
    
    # Custom side effect for Update.de_json to return dynamic mock
    def mock_de_json(data, bot):
        m = MagicMock()
        m.update_id = data.get('update_id')
        if 'message' in data:
            msg_data = data['message']
            m.message.message_id = msg_data.get('message_id')
            m.message.chat_id = msg_data['chat']['id']  # Note: helper property usually
            m.message.chat.id = msg_data['chat']['id']
            m.message.from_user.id = msg_data['from']['id']
            m.message.text = msg_data.get('text')
            
            # Allow accessing unknown attributes as None/Mock
        return m

    # Apply patches
    with patch.object(tgbot_module, 'bot_instance', bot_mock), \
         patch.object(tgbot_module, 'admin_manager', admin_mock), \
         patch.object(tgbot_module, 'telegram_app', tg_mock), \
         patch('src.bot.tgbot.Update.de_json', side_effect=mock_de_json):
        
        yield {
            'bot': bot_mock,
            'admin': admin_mock,
            'tg': tg_mock
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
    
    # Should call chat
    mock_deps['bot'].chat.assert_called_with("Hello", respond=True)
    mock_deps['tg'].bot.send_message.assert_called()

def test_webhook_whitelist_block(client, mock_deps):
    """Test blocking when chat is not in whitelist."""
    # Setup whitelist
    mock_deps['admin'].config.allowed_chats = [999]
    
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

def test_webhook_frequency_limit(client, mock_deps):
    """Test frequency limiting."""
    mock_deps['admin'].config.response_frequency = 2
    mock_deps['admin'].config.allowed_chats = [123] # Allow chat
    
    chat_id = 123
    
    # Reset counters
    tgbot_module.chat_counters = {}
    
    update = {
        "update_id": 3,
        "message": {
            "message_id": 3,
            "date": 123456,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Test"},
            "text": "Msg 1"
        }
    }
    
    # Message 1: 1 % 2 != 0 -> No response
    client.post("/webhook", json=update)
    mock_deps['bot'].chat.assert_called_with("Msg 1", respond=False)
    
    # Message 2: 2 % 2 == 0 -> Response
    update["message"]["text"] = "Msg 2"
    client.post("/webhook", json=update)
    mock_deps['bot'].chat.assert_called_with("Msg 2", respond=True)
