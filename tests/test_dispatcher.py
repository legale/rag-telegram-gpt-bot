"""Tests for src/core/dispatcher.py"""

import pytest
from unittest.mock import Mock, AsyncMock
from abc import ABC
from src.core.dispatcher import (
    CommandContext,
    CommandResult,
    CommandHandler,
    AsyncCommandHandler,
    CommandDispatcher
)


class TestCommandContext:
    """Tests for CommandContext"""
    
    def test_init_with_all_params(self):
        """Test CommandContext initialization with all parameters"""
        context = CommandContext(
            user_id="123",
            chat_id="456",
            command_name="test",
            args=["arg1", "arg2"],
            metadata={"key": "value"}
        )
        
        assert context.user_id == "123"
        assert context.chat_id == "456"
        assert context.command_name == "test"
        assert context.args == ["arg1", "arg2"]
        assert context.metadata == {"key": "value"}
    
    def test_init_with_minimal_params(self):
        """Test CommandContext initialization with minimal parameters"""
        context = CommandContext(command_name="test")
        
        assert context.user_id is None
        assert context.chat_id is None
        assert context.command_name == "test"
        assert context.args == []
        assert context.metadata == {}
    
    def test_post_init_defaults_args(self):
        """Test CommandContext __post_init__ sets default args"""
        context = CommandContext(command_name="test", args=None)
        
        assert context.args == []
    
    def test_post_init_defaults_metadata(self):
        """Test CommandContext __post_init__ sets default metadata"""
        context = CommandContext(command_name="test", metadata=None)
        
        assert context.metadata == {}


class TestCommandResult:
    """Tests for CommandResult"""
    
    def test_init_with_all_params(self):
        """Test CommandResult initialization with all parameters"""
        result = CommandResult(
            success=True,
            message="Test message",
            data={"key": "value"},
            error="Test error"
        )
        
        assert result.success is True
        assert result.message == "Test message"
        assert result.data == {"key": "value"}
        assert result.error == "Test error"
    
    def test_init_with_minimal_params(self):
        """Test CommandResult initialization with minimal parameters"""
        result = CommandResult(success=True, message="Test")
        
        assert result.success is True
        assert result.message == "Test"
        assert result.data == {}
        assert result.error is None
    
    def test_post_init_defaults_data(self):
        """Test CommandResult __post_init__ sets default data"""
        result = CommandResult(success=True, message="Test", data=None)
        
        assert result.data == {}


class TestCommandHandler:
    """Tests for CommandHandler abstract base class"""
    
    def test_is_abstract(self):
        """Test that CommandHandler is abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            CommandHandler()
    
    def test_can_be_subclassed(self):
        """Test that CommandHandler can be subclassed"""
        class ConcreteHandler(CommandHandler):
            def handle(self, context):
                return CommandResult(success=True, message="OK")
        
        handler = ConcreteHandler()
        assert isinstance(handler, CommandHandler)
        assert isinstance(handler, ABC)
        
        context = CommandContext(command_name="test")
        result = handler.handle(context)
        assert result.success is True


class TestAsyncCommandHandler:
    """Tests for AsyncCommandHandler abstract base class"""
    
    def test_is_abstract(self):
        """Test that AsyncCommandHandler is abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            AsyncCommandHandler()
    
    @pytest.mark.asyncio
    async def test_can_be_subclassed(self):
        """Test that AsyncCommandHandler can be subclassed"""
        class ConcreteAsyncHandler(AsyncCommandHandler):
            async def handle(self, context):
                return CommandResult(success=True, message="OK")
        
        handler = ConcreteAsyncHandler()
        assert isinstance(handler, AsyncCommandHandler)
        assert isinstance(handler, ABC)
        
        context = CommandContext(command_name="test")
        result = await handler.handle(context)
        assert result.success is True


class TestCommandDispatcher:
    """Tests for CommandDispatcher"""
    
    def test_init(self):
        """Test CommandDispatcher initialization"""
        dispatcher = CommandDispatcher()
        
        assert dispatcher.handlers == {}
        assert dispatcher.async_handlers == {}
    
    def test_register_sync_handler(self):
        """Test registering a synchronous handler"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        
        dispatcher.register("test", handler)
        
        assert dispatcher.handlers["test"] == handler
        assert "test" not in dispatcher.async_handlers
    
    def test_register_sync_handler_normalizes_name(self):
        """Test that register normalizes command name"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        
        dispatcher.register("/TEST", handler)
        
        assert dispatcher.handlers["test"] == handler
    
    def test_register_async_handler(self):
        """Test registering an asynchronous handler"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=AsyncCommandHandler)
        
        dispatcher.register_async("test", handler)
        
        assert dispatcher.async_handlers["test"] == handler
        assert "test" not in dispatcher.handlers
    
    def test_register_async_handler_normalizes_name(self):
        """Test that register_async normalizes command name"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=AsyncCommandHandler)
        
        dispatcher.register_async("/TEST", handler)
        
        assert dispatcher.async_handlers["test"] == handler
    
    def test_dispatch_success(self):
        """Test successful command dispatch"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        handler.handle.return_value = CommandResult(success=True, message="OK")
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="test")
        result = dispatcher.dispatch(context)
        
        assert result.success is True
        assert result.message == "OK"
        handler.handle.assert_called_once_with(context)
    
    def test_dispatch_normalizes_command_name(self):
        """Test that dispatch normalizes command name"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        handler.handle.return_value = CommandResult(success=True, message="OK")
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="/TEST")
        result = dispatcher.dispatch(context)
        
        assert result.success is True
        handler.handle.assert_called_once()
    
    def test_dispatch_command_not_found(self):
        """Test dispatch when command is not found"""
        dispatcher = CommandDispatcher()
        context = CommandContext(command_name="unknown")
        
        result = dispatcher.dispatch(context)
        
        assert result.success is False
        assert "Неизвестная команда" in result.message
        assert result.error == "Command 'unknown' not found"
    
    def test_dispatch_handler_exception(self):
        """Test dispatch when handler raises exception"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        handler.handle.side_effect = Exception("Handler error")
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="test")
        result = dispatcher.dispatch(context)
        
        assert result.success is False
        assert "Ошибка при выполнении команды" in result.message
        assert result.error == "Handler error"
    
    @pytest.mark.asyncio
    async def test_dispatch_async_success(self):
        """Test successful async command dispatch"""
        dispatcher = CommandDispatcher()
        handler = AsyncMock()
        handler.handle = AsyncMock(return_value=CommandResult(success=True, message="OK"))
        dispatcher.register_async("test", handler)
        
        context = CommandContext(command_name="test")
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is True
        assert result.message == "OK"
        handler.handle.assert_called_once_with(context)
    
    @pytest.mark.asyncio
    async def test_dispatch_async_fallback_to_sync(self):
        """Test async dispatch falls back to sync handler"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        handler.handle.return_value = CommandResult(success=True, message="OK")
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="test")
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is True
        assert result.message == "OK"
        handler.handle.assert_called_once_with(context)
    
    @pytest.mark.asyncio
    async def test_dispatch_async_prefers_async_handler(self):
        """Test async dispatch prefers async handler over sync"""
        dispatcher = CommandDispatcher()
        sync_handler = Mock()
        async_handler = AsyncMock()
        async_handler.handle = AsyncMock(return_value=CommandResult(success=True, message="Async OK"))
        sync_handler.handle = Mock(return_value=CommandResult(success=True, message="Sync OK"))
        
        dispatcher.register("test", sync_handler)
        dispatcher.register_async("test", async_handler)
        
        context = CommandContext(command_name="test")
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is True
        assert result.message == "Async OK"
        async_handler.handle.assert_called_once_with(context)
        sync_handler.handle.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_dispatch_async_handler_exception(self):
        """Test async dispatch when handler raises exception"""
        dispatcher = CommandDispatcher()
        handler = AsyncMock(spec=AsyncCommandHandler)
        handler.handle.side_effect = Exception("Async handler error")
        dispatcher.register_async("test", handler)
        
        context = CommandContext(command_name="test")
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is False
        assert "Ошибка при выполнении команды" in result.message
        assert result.error == "Async handler error"
    
    @pytest.mark.asyncio
    async def test_dispatch_async_sync_fallback_exception(self):
        """Test async dispatch when sync fallback raises exception"""
        dispatcher = CommandDispatcher()
        handler = Mock(spec=CommandHandler)
        handler.handle.side_effect = Exception("Sync handler error")
        dispatcher.register("test", handler)
        
        context = CommandContext(command_name="test")
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is False
        assert "Ошибка при выполнении команды" in result.message
        assert result.error == "Sync handler error"
    
    @pytest.mark.asyncio
    async def test_dispatch_async_command_not_found(self):
        """Test async dispatch when command is not found"""
        dispatcher = CommandDispatcher()
        context = CommandContext(command_name="unknown")
        
        result = await dispatcher.dispatch_async(context)
        
        assert result.success is False
        assert "Неизвестная команда" in result.message
        assert result.error == "Command 'unknown' not found"

