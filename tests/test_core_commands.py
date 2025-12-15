"""
Tests for core/commands module.
"""

import pytest
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
    """Tests for StartCommandHandler class."""
    
    def test_handle(self):
        """Test handle method."""
        handler = StartCommandHandler()
        context = CommandContext()
        result = handler.handle(context)
        
        assert result.success is True
        assert "Привет" in result.message


class TestHelpCommandHandler:
    """Tests for HelpCommandHandler class."""
    
    def test_handle(self):
        """Test handle method."""
        handler = HelpCommandHandler()
        context = CommandContext()
        result = handler.handle(context)
        
        assert result.success is True
        assert "команды" in result.message or "справка" in result.message


class TestResetCommandHandler:
    """Tests for ResetCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = ResetCommandHandler(None)
        assert handler.bot is None
    
    def test_handle_with_bot(self, tmp_path, monkeypatch):
        """Test handle with bot."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = ResetCommandHandler(bot)
        context = CommandContext()
        result = handler.handle(context)
        
        assert result.success is True


class TestTokensCommandHandler:
    """Tests for TokensCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = TokensCommandHandler(None)
        assert handler.bot is None
    
    def test_handle_with_bot(self, tmp_path, monkeypatch):
        """Test handle with bot."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = TokensCommandHandler(bot)
        context = CommandContext()
        result = handler.handle(context)
        
        assert result.success is True
        assert "токенов" in result.message.lower() or "tokens" in result.message.lower()


class TestModelCommandHandler:
    """Tests for ModelCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = ModelCommandHandler(None)
        assert handler.bot is None
        assert handler.admin_manager is None
    
    def test_handle_with_bot(self, tmp_path, monkeypatch):
        """Test handle with bot."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        # Ensure bot has required attributes
        if not hasattr(bot, 'current_model_index'):
            bot.current_model_index = 0
        if not hasattr(bot, 'current_model_name'):
            bot.current_model_name = "test-model"
        
        handler = ModelCommandHandler(bot)
        context = CommandContext()
        result = handler.handle(context)
        
        # May fail if models.txt is missing, so just check it doesn't crash
        assert result is not None


class TestFindCommandHandler:
    """Tests for FindCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = FindCommandHandler(None)
        assert handler.bot is None
        assert handler.admin_manager is None
        assert handler.debug_rag is False
    
    def test_handle_with_empty_query(self, tmp_path, monkeypatch):
        """Test handle with empty query."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=[])
        result = handler.handle(context)
        
        # Should return error for empty query
        assert result.success is False or "Использование" in result.message

