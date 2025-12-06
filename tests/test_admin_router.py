"""
Tests for Admin Router.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.bot.admin_router import AdminCommandRouter

@pytest.fixture
def mock_router():
    router = AdminCommandRouter()
    return router

@pytest.mark.asyncio
async def test_route_unauthorized(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin"
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = False
    
    res = await mock_router.route(update, context, admin_manager)
    assert "только администратору" in res

@pytest.mark.asyncio
async def test_route_main_menu(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin"
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = True
    
    res = await mock_router.route(update, context, admin_manager)
    assert "Панель администратора" in res

@pytest.mark.asyncio
async def test_route_command(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin test"
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = True
    
    handler = AsyncMock(return_value="OK")
    mock_router.register("test", handler)
    
    res = await mock_router.route(update, context, admin_manager)
    assert "OK" in res
    handler.assert_called_with(update, context, admin_manager, [])

@pytest.mark.asyncio
async def test_route_subcommand(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin sub cmd arg1"
    context = MagicMock()
    
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = True
    
    handler = AsyncMock(return_value="OK Sub")
    mock_router.register("sub", handler, subcommand="cmd")
    
    res = await mock_router.route(update, context, admin_manager)
    assert "OK Sub" in res
    handler.assert_called_with(update, context, admin_manager, ['arg1'])

@pytest.mark.asyncio
async def test_route_unknown(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin unknown"
    context = MagicMock()
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = True
    
    res = await mock_router.route(update, context, admin_manager)
    assert "Неизвестная команда" in res

@pytest.mark.asyncio
async def test_route_missing_subcommand(mock_router):
    update = MagicMock()
    update.message.from_user.id = 1
    update.message.text = "/admin sub"
    context = MagicMock()
    admin_manager = MagicMock()
    admin_manager.is_admin.return_value = True
    
    handler = AsyncMock()
    mock_router.register("sub", handler, subcommand="cmd")
    
    res = await mock_router.route(update, context, admin_manager)
    assert "требует подкоманду" in res
