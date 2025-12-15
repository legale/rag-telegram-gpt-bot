"""
Tests for core/admin_commands module.
"""

import pytest
from src.core.admin_commands import AdminSetCommandHandler, AdminGetCommandHandler, AdminCommandHandler
from src.core.dispatcher import CommandContext, CommandResult


class TestAdminSetCommandHandler:
    """Tests for AdminSetCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = AdminSetCommandHandler(None)
        assert handler.admin_manager is None
    
    def test_handle_without_admin_manager(self):
        """Test handle without admin manager."""
        handler = AdminSetCommandHandler(None)
        context = CommandContext()
        result = handler.handle(context)
        
        # Should be async, but test sync call
        import asyncio
        result = asyncio.run(handler.handle(context))
        assert result.success is False
        assert "недоступна" in result.message
    
    def test_handle_without_update(self):
        """Test handle without update in metadata."""
        class MockAdminManager:
            pass
        
        handler = AdminSetCommandHandler(MockAdminManager())
        context = CommandContext()
        
        import asyncio
        result = asyncio.run(handler.handle(context))
        assert result.success is False


class TestAdminGetCommandHandler:
    """Tests for AdminGetCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = AdminGetCommandHandler(None)
        assert handler.admin_manager is None
    
    def test_handle_without_admin_manager(self):
        """Test handle without admin manager."""
        handler = AdminGetCommandHandler(None)
        context = CommandContext()
        
        import asyncio
        result = asyncio.run(handler.handle(context))
        assert result.success is False
    
    def test_handle_without_user_id(self):
        """Test handle without user_id."""
        class MockAdminManager:
            pass
        
        handler = AdminGetCommandHandler(MockAdminManager())
        context = CommandContext(user_id=None)
        
        import asyncio
        result = asyncio.run(handler.handle(context))
        assert result.success is False


class TestAdminCommandHandler:
    """Tests for AdminCommandHandler class."""
    
    def test_init(self):
        """Test initialization."""
        handler = AdminCommandHandler(None)
        assert handler.admin_router is None
    
    def test_handle_without_admin_router(self):
        """Test handle without admin router."""
        handler = AdminCommandHandler(None)
        context = CommandContext()
        
        import asyncio
        result = asyncio.run(handler.handle(context))
        assert result.success is False

