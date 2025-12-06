"""
Tests for Admin Commands.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.bot.admin_commands import SettingsCommands, ControlCommands, HelpCommands

@pytest.fixture
def mock_context():
    # Helper to create update/context/manager mocks
    update = MagicMock()
    update.message.chat_id = 123
    update.message.from_user.id = 1
    
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.config = MagicMock()
    admin_manager.config.allowed_chats = []
    admin_manager.config.response_frequency = 1
    
    profile_manager = MagicMock()
    
    return update, context, admin_manager, profile_manager

class TestSettingsCommands:
    @pytest.mark.asyncio
    async def test_manage_chats_list(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Empty list
        response = await settings.manage_chats(update, context, admin_manager, ['list'])
        assert "список разрешенных чатов пуст" in response.lower()
        
        # With chats
        admin_manager.config.allowed_chats = [123, 456]
        response = await settings.manage_chats(update, context, admin_manager, ['list'])
        assert "123" in response
        assert "456" in response

    @pytest.mark.asyncio
    async def test_manage_chats_add(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Simulate 'in' operator on mock list by setting it to real list initially
        # admin_manager.config.allowed_chats is a Mock, but we can set it to list
        
        # Test 1: Add current chat (123)
        admin_manager.config.allowed_chats = []
        response = await settings.manage_chats(update, context, admin_manager, ['add'])
        assert "добавлен" in response
        admin_manager.config.add_allowed_chat.assert_called_with(123)
        
        # Test 2: Add explicit chat (999)
        response = await settings.manage_chats(update, context, admin_manager, ['add', '999'])
        assert "добавлен" in response
        admin_manager.config.add_allowed_chat.assert_called_with(999)
        
        # Test 3: Add duplicate
        admin_manager.config.allowed_chats = [123]
        response = await settings.manage_chats(update, context, admin_manager, ['add'])
        assert "уже в списке" in response.lower()

    @pytest.mark.asyncio
    async def test_manage_frequency(self, mock_context):
        update, context, admin_manager, pm = mock_context
        settings = SettingsCommands(pm)
        
        # Get
        response = await settings.manage_frequency(update, context, admin_manager, [])
        assert "1 ответ" in response
        
        # Set valid
        response = await settings.manage_frequency(update, context, admin_manager, ['5'])
        assert "5" in response
        assert admin_manager.config.response_frequency == 5
        
        # Set invalid
        response = await settings.manage_frequency(update, context, admin_manager, ['abc'])
        assert "используйте число" in response.lower()

class TestControlCommands:
    @pytest.mark.asyncio
    async def test_restart_bot(self, mock_context):
        update, context, admin_manager, pm = mock_context
        # Force no job queue to test asyncio fallback
        context.job_queue = None
        
        control = ControlCommands(pm)
        
        # We need to mock asyncio.create_task
        with patch('asyncio.create_task') as mock_task:
            response = await control.restart_bot(update, context, admin_manager, [])
            assert "перезапуск" in response.lower()
            mock_task.assert_called()
            
            # Close the coroutine to avoid RuntimeWarning
            # The coroutine _delayed_restart is created but never awaited by mocked create_task
            coro = mock_task.call_args[0][0]
            coro.close()

class TestHelpCommands:
    @pytest.mark.asyncio
    async def test_show_help(self, mock_context):
        update, context, admin_manager, pm = mock_context
        helper = HelpCommands()
        
        response = await helper.show_help(update, context, admin_manager, [])
        assert "справка по админ-командам" in response.lower()
