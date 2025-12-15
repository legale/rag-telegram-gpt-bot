"""
Tests for core/dispatcher module.
"""

import pytest
from src.core.dispatcher import (
    CommandDispatcher,
    CommandContext,
    CommandResult,
    CommandHandler,
    AsyncCommandHandler
)


class TestCommandContext:
    """Tests for CommandContext dataclass."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        context = CommandContext()
        assert context.user_id is None
        assert context.chat_id is None
        assert context.command_name == ""
        assert context.args == []
        assert context.metadata == {}
    
    def test_init_with_values(self):
        """Test initialization with values."""
        context = CommandContext(
            user_id="user1",
            chat_id="chat1",
            command_name="/start",
            args=["arg1"],
            metadata={"key": "value"}
        )
        assert context.user_id == "user1"
        assert context.chat_id == "chat1"
        assert context.command_name == "/start"
        assert context.args == ["arg1"]
        assert context.metadata == {"key": "value"}


class TestCommandResult:
    """Tests for CommandResult dataclass."""
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        result = CommandResult(success=True, message="test")
        assert result.success is True
        assert result.message == "test"
        assert result.data == {}
        assert result.error is None
    
    def test_init_with_data(self):
        """Test initialization with data."""
        result = CommandResult(
            success=True,
            message="test",
            data={"key": "value"},
            error="error"
        )
        assert result.data == {"key": "value"}
        assert result.error == "error"


class TestCommandHandler:
    """Tests for CommandHandler abstract class."""
    
    def test_is_abstract(self):
        """Test that CommandHandler is abstract."""
        with pytest.raises(TypeError):
            CommandHandler()


class TestAsyncCommandHandler:
    """Tests for AsyncCommandHandler abstract class."""
    
    def test_is_abstract(self):
        """Test that AsyncCommandHandler is abstract."""
        with pytest.raises(TypeError):
            AsyncCommandHandler()


class TestCommandDispatcher:
    """Tests for CommandDispatcher class."""
    
    def test_init(self):
        """Test initialization."""
        dispatcher = CommandDispatcher()
        assert dispatcher.handlers == {}
        assert dispatcher.async_handlers == {}
    
    def test_register(self):
        """Test register method."""
        class TestHandler(CommandHandler):
            def handle(self, context):
                return CommandResult(success=True, message="test")
        
        dispatcher = CommandDispatcher()
        handler = TestHandler()
        dispatcher.register("test", handler)
        
        assert "test" in dispatcher.handlers
        assert dispatcher.handlers["test"] == handler
    
    def test_register_normalizes_name(self):
        """Test register normalizes command name."""
        class TestHandler(CommandHandler):
            def handle(self, context):
                return CommandResult(success=True, message="test")
        
        dispatcher = CommandDispatcher()
        handler = TestHandler()
        dispatcher.register("/TEST", handler)
        
        assert "test" in dispatcher.handlers
    
    def test_register_async(self):
        """Test register_async method."""
        class TestAsyncHandler(AsyncCommandHandler):
            async def handle(self, context):
                return CommandResult(success=True, message="test")
        
        dispatcher = CommandDispatcher()
        handler = TestAsyncHandler()
        dispatcher.register_async("test", handler)
        
        assert "test" in dispatcher.async_handlers
        assert dispatcher.async_handlers["test"] == handler
    
    def test_dispatch_unknown_command(self):
        """Test dispatch with unknown command."""
        dispatcher = CommandDispatcher()
        context = CommandContext(command_name="/unknown")
        result = dispatcher.dispatch(context)
        
        assert result.success is False
        assert "Неизвестная команда" in result.message
    
    def test_dispatch_known_command(self):
        """Test dispatch with known command."""
        class TestHandler(CommandHandler):
            def handle(self, context):
                return CommandResult(success=True, message="success")
        
        dispatcher = CommandDispatcher()
        handler = TestHandler()
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="/test")
        result = dispatcher.dispatch(context)
        
        assert result.success is True
        assert result.message == "success"
    
    def test_dispatch_handler_exception(self):
        """Test dispatch when handler raises exception."""
        class FailingHandler(CommandHandler):
            def handle(self, context):
                raise ValueError("Test error")
        
        dispatcher = CommandDispatcher()
        handler = FailingHandler()
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="/test")
        result = dispatcher.dispatch(context)
        
        assert result.success is False
        assert "Ошибка" in result.message
    
    def test_dispatch_async_unknown_command(self):
        """Test dispatch_async with unknown command."""
        import asyncio
        dispatcher = CommandDispatcher()
        context = CommandContext(command_name="/unknown")
        result = asyncio.run(dispatcher.dispatch_async(context))
        
        assert result.success is False
        assert "Неизвестная команда" in result.message
    
    def test_dispatch_async_known_command(self):
        """Test dispatch_async with known async command."""
        import asyncio
        class TestAsyncHandler(AsyncCommandHandler):
            async def handle(self, context):
                return CommandResult(success=True, message="success")
        
        dispatcher = CommandDispatcher()
        handler = TestAsyncHandler()
        dispatcher.register_async("test", handler)
        
        context = CommandContext(command_name="/test")
        result = asyncio.run(dispatcher.dispatch_async(context))
        
        assert result.success is True
        assert result.message == "success"
    
    def test_dispatch_async_fallback_to_sync(self):
        """Test dispatch_async falls back to sync handler."""
        import asyncio
        class TestHandler(CommandHandler):
            def handle(self, context):
                return CommandResult(success=True, message="sync")
        
        dispatcher = CommandDispatcher()
        handler = TestHandler()
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="/test")
        result = asyncio.run(dispatcher.dispatch_async(context))
        
        assert result.success is True
        assert result.message == "sync"

