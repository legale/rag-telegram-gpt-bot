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
    
    def test_handle_no_args_shows_help(self, tmp_path, monkeypatch):
        """Test handle shows help when no args provided."""
        handler = self._create_handler(tmp_path, monkeypatch)
        context = CommandContext(args=[])
        result = handler.handle(context)
        assert result.success is True
        assert "Команды управления моделью" in result.message

    def test_handle_help(self, tmp_path, monkeypatch):
        """Test handle help subcommand."""
        handler = self._create_handler(tmp_path, monkeypatch)
        context = CommandContext(args=["help"])
        result = handler.handle(context)
        assert result.success is True
        assert "Команды управления моделью" in result.message

    def test_handle_list(self, tmp_path, monkeypatch):
        """Test handle list subcommand."""
        handler = self._create_handler(tmp_path, monkeypatch)
        handler.bot.available_models = ["model1", "model2"]
        # bot.current_model_name reads from self.llm_client.model_name
        handler.bot.llm_client.model_name = "model1"
        handler.bot.model_max_tokens = {"model1": 140000, "model2": 10000}
        
        context = CommandContext(args=["list"])
        result = handler.handle(context)
        
        assert result.success is True
        assert "Доступные модели:" in result.message
        assert "model1" in result.message
        assert "model2" in result.message
        assert "(текущая)" in result.message

    def test_handle_get(self, tmp_path, monkeypatch):
        """Test handle get subcommand."""
        handler = self._create_handler(tmp_path, monkeypatch)
        # Mock get_current_model method
        handler.bot.get_current_model = lambda: "Current model info"
        
        context = CommandContext(args=["get"])
        result = handler.handle(context)
        
        assert result.success is True
        assert "Current model info" in result.message

    def test_handle_set_success(self, tmp_path, monkeypatch):
        """Test handle set subcommand success."""
        handler = self._create_handler(tmp_path, monkeypatch)
        
        # We need to ensure that when set_model is called, it updates the state 
        # that current_model_name reads from, OR we mock set_model to do nothing 
        # and manually force the state for verification.
        
        def mock_set_model(name):
            handler.bot.llm_client.model_name = name
            return f"Model set to {name}"
            
        handler.bot.set_model = mock_set_model
        
        context = CommandContext(args=["set", "new-model"])
        result = handler.handle(context)
        
        assert result.success is True
        assert "Model set to new-model" in result.message
        assert result.data["model"] == "new-model"

    def test_handle_set_missing_arg(self, tmp_path, monkeypatch):
        """Test handle set subcommand without model name."""
        handler = self._create_handler(tmp_path, monkeypatch)
        context = CommandContext(args=["set"])
        result = handler.handle(context)
        
        assert result.success is False
        assert "Укажите имя модели" in result.message

    def _create_handler(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        return ModelCommandHandler(bot)


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
    
    def test_handle_with_list_action(self, tmp_path, monkeypatch):
        """Test handle with list action."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["hybrid", "list"])
        result = handler.handle(context)
        
        assert result.success is True
        assert "Доступные методы RAG" in result.message
        assert result.data["action"] == "list"
        assert result.data["rag_method"] == "hybrid"
    
    def test_handle_with_hybrid_search(self, tmp_path, monkeypatch):
        """Test handle with hybrid search method."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["hybrid", "test", "query"])
        result = handler.handle(context)
        
        # Should succeed (even if no results found)
        # The key is that it doesn't crash with AttributeError
        assert result is not None
        # Either success with empty results or error message, but not AttributeError
        if result.success:
            assert "message_parts_list" in result.data or "results_count" in result.data
        else:
            # Error should not be about missing attributes
            assert "vector_db_path" not in str(result.error).lower()
            assert "db_url" not in str(result.error).lower()
    
    def test_handle_with_vector_only_search(self, tmp_path, monkeypatch):
        """Test handle with vector_only search method."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["vector_only", "test", "query"])
        result = handler.handle(context)
        
        # Should succeed (even if no results found)
        assert result is not None
        if result.success:
            assert result.data["rag_method"] == "vector_only"
        else:
            # Error should not be about missing attributes
            assert "vector_db_path" not in str(result.error).lower()
    
    def test_handle_with_fts_only_search(self, tmp_path, monkeypatch):
        """Test handle with fts_only search method."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["fts_only", "test", "query"])
        result = handler.handle(context)
        
        # Should succeed (even if no results found)
        assert result is not None
        if result.success:
            assert result.data["rag_method"] == "fts_only"
        else:
            # Error should not be about missing attributes
            assert "vector_db_path" not in str(result.error).lower()
    
    def test_handle_with_invalid_rag_method(self, tmp_path, monkeypatch):
        """Test handle with invalid RAG method."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(tmp_path / "vector"),
            model_name="test-model"
        )
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["invalid_method", "query"])
        result = handler.handle(context)
        
        assert result.success is False
        assert "Неизвестный метод RAG" in result.message
    
    def test_handle_uses_vector_store_persist_directory(self, tmp_path, monkeypatch):
        """Test that handler correctly uses vector_store.persist_directory."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from src.bot.core import LegaleBot
        
        vector_path = tmp_path / "vector"
        bot = LegaleBot(
            db_url=f"sqlite:///{tmp_path}/test.db",
            vector_db_path=str(vector_path),
            model_name="test-model"
        )
        
        # Verify bot has vector_store with persist_directory
        assert hasattr(bot, 'vector_store')
        assert hasattr(bot.vector_store, 'persist_directory')
        assert bot.vector_store.persist_directory == str(vector_path)
        
        handler = FindCommandHandler(bot)
        context = CommandContext(args=["hybrid", "test"])
        result = handler.handle(context)
        
        # Should not crash with AttributeError about vector_db_path
        assert result is not None
        if not result.success:
            assert "vector_db_path" not in str(result.error).lower()

