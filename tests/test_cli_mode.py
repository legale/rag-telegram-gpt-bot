
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.app.main_cli import handle_command_async, handle_command, CommandContext, CommandDispatcher
from src.bot.admin_router import AdminCommandRouter
from src.bot.admin import AdminManager
from pathlib import Path

class TestCliMode:
    """Tests for CLI mode command execution."""

    @pytest.fixture
    def mock_dispatcher(self):
        dispatcher = Mock(spec=CommandDispatcher)
        dispatcher.dispatch_async = AsyncMock()
        dispatcher.dispatch = Mock()
        return dispatcher

    @pytest.mark.asyncio
    async def test_handle_command_async_admin(self, mock_dispatcher):
        """Test handling admin command in async mode (CLI style)."""
        mock_dispatcher.dispatch_async.return_value = Mock(success=True, message="Admin Response", error=None, data={})
        
        result_message, result_data = await handle_command_async(
            command="/admin allowed list",
            dispatcher=mock_dispatcher,
            user_id="0",
            chat_id="0"
        )
        
        assert result_message == "Admin Response"
        mock_dispatcher.dispatch_async.assert_called_once()
        args = mock_dispatcher.dispatch_async.call_args[0][0]
        assert isinstance(args, CommandContext)
        assert args.command_name == "/admin"
        assert args.args == ["allowed", "list"]

    @pytest.mark.asyncio
    async def test_handle_command_async_model(self, mock_dispatcher):
        """Test handling model command in async mode wrapper (CLI style)."""
        # dispatcher.dispatch (sync) is called for non-admin commands
        mock_dispatcher.dispatch.return_value = Mock(success=True, message="Model List", error=None, data={})
        
        result_message, result_data = await handle_command_async(
            command="/model list",
            dispatcher=mock_dispatcher,
            user_id="0",
            chat_id="0"
        )
        
        assert result_message == "Model List"
        mock_dispatcher.dispatch.assert_called_once()
        mock_dispatcher.dispatch_async.assert_not_called() 
        args = mock_dispatcher.dispatch.call_args[0][0]
        assert args.command_name == "/model"
        assert args.args == ["list"]

    @pytest.mark.asyncio
    async def test_handle_command_async_normalize_set_admin(self, mock_dispatcher):
        """Test normalization of /set_admin to /admin_set."""
        mock_dispatcher.dispatch_async.return_value = Mock(success=True, message="Set Admin", error=None, data={})
        
        await handle_command_async(
            command="/set_admin password",
            dispatcher=mock_dispatcher
        )
        
        args = mock_dispatcher.dispatch_async.call_args[0][0]
        assert args.command_name == "/admin_set"

    @pytest.mark.asyncio
    async def test_handle_command_async_unknown_command(self, mock_dispatcher):
        """Test handling unknown command format (no slash)."""
        result_message, result_data = await handle_command_async(
            command="hello world",
            dispatcher=mock_dispatcher
        )
        
        assert result_message is None
        assert result_data is None
        mock_dispatcher.dispatch.assert_not_called()
        mock_dispatcher.dispatch_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_command_async_empty(self, mock_dispatcher):
        """Test handling empty command."""
        result_message, result_data = await handle_command_async(
            command="",
            dispatcher=mock_dispatcher
        )
        assert result_message is None

    def test_handle_command_sync(self, mock_dispatcher):
        """Test handling command via sync handle_command (deprecated but checked)."""
        mock_dispatcher.dispatch.return_value = Mock(success=True, message="Sync Response", error=None, data={})
        
        result = handle_command(
            command="/help",
            dispatcher=mock_dispatcher
        )
        
        assert result == "Sync Response"
        mock_dispatcher.dispatch.assert_called_once()


class TestBotModeIntegration:
    """Tests simulating Bot mode command routing."""
    
    @pytest.mark.asyncio
    async def test_admin_commands_route_correctly(self):
        """Verify that admin commands are correctly routed in bot mode context."""
        # This duplicates some logic from test_tgbot but focuses on the router integraion
        
        mock_admin_manager = Mock(spec=AdminManager)
        mock_admin_manager.is_admin.return_value = True
        
        router = AdminCommandRouter()
        # Mock handlers for the router
        mock_handler = AsyncMock(return_value="Handler Result")
        router.register("test", mock_handler)
        
        # Manually invoke route
        update = Mock()
        update.message.from_user.id = 123
        update.message.text = "/admin test"
        
        response = await router.route(update, None, mock_admin_manager)
        
        assert response == "Handler Result"
        mock_handler.assert_called_once()

