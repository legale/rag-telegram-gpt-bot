"""
Tests for main_cli module.
"""

import pytest
from src.app.main_cli import create_dispatcher, handle_command
from src.core.dispatcher import CommandDispatcher, CommandContext
from src.bot.core import LegaleBot
from src.bot.admin import AdminManager
from src.bot.admin_router import AdminCommandRouter


class TestCreateDispatcher:
    """Tests for create_dispatcher function."""
    
    def test_create_with_bot_only(self, tmp_path, monkeypatch):
        """Test creating dispatcher with bot only."""
        # Mock environment variables to avoid API key requirement
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        dispatcher = create_dispatcher(bot)
        assert dispatcher is not None
        assert isinstance(dispatcher, CommandDispatcher)
    
    def test_create_with_admin_manager(self, tmp_path, monkeypatch):
        """Test creating dispatcher with admin manager."""
        # Mock environment variables to avoid API key requirement
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        admin_manager = AdminManager(str(profile_dir))
        
        dispatcher = create_dispatcher(bot, admin_manager=admin_manager)
        assert dispatcher is not None


class TestHandleCommand:
    """Tests for handle_command function."""
    
    def test_handle_non_command(self):
        """Test handling non-command text."""
        dispatcher = CommandDispatcher()
        result = handle_command("not a command", dispatcher)
        assert result is None
    
    def test_handle_command_without_slash(self):
        """Test handling text without slash."""
        dispatcher = CommandDispatcher()
        result = handle_command("help", dispatcher)
        assert result is None
    
    def test_handle_unknown_command(self):
        """Test handling unknown command."""
        dispatcher = CommandDispatcher()
        result = handle_command("/unknown", dispatcher)
        # Unknown command returns error message, not None
        assert result is not None
        assert "Неизвестная команда" in result

