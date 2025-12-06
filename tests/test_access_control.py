"""
Tests for AccessControlService.

Tests cover:
- Admin access checks
- Private chat access logic
- Group chat access logic
- Command vs text message logic
- Whitelist checks
- Access denial messages
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.bot.utils.access_control import AccessControlService


class TestIsAdmin:
    """Tests for is_admin() method."""
    
    def test_is_admin_true(self):
        """Test admin check returns True for admin user."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = True
        
        service = AccessControlService(admin_manager)
        assert service.is_admin(12345) is True
        admin_manager.is_admin.assert_called_once_with(12345)
    
    def test_is_admin_false(self):
        """Test admin check returns False for non-admin user."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        
        service = AccessControlService(admin_manager)
        assert service.is_admin(67890) is False
        admin_manager.is_admin.assert_called_once_with(67890)
    
    def test_is_admin_no_manager(self):
        """Test admin check returns False when admin_manager is None."""
        service = AccessControlService(None)
        assert service.is_admin(12345) is False


class TestIsAllowed:
    """Tests for is_allowed() method."""
    
    def test_admin_always_allowed_private(self):
        """Test admin is always allowed in private chat."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = True
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=12345,
            chat_id=12345,
            is_private=True,
            is_command=False
        )
        
        assert allowed is True
        assert reason is None
    
    def test_admin_always_allowed_group(self):
        """Test admin is always allowed in group chat."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = True
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=12345,
            chat_id=-100123456,
            is_private=False,
            is_command=False
        )
        
        assert allowed is True
        assert reason is None
    
    def test_private_non_admin_denied(self):
        """Test non-admin is denied in private chat."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=67890,
            is_private=True,
            is_command=False
        )
        
        assert allowed is False
        assert reason == "private_non_admin"
    
    def test_private_non_admin_command_denied(self):
        """Test non-admin command is denied in private chat."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=67890,
            is_private=True,
            is_command=True
        )
        
        assert allowed is False
        assert reason == "private_non_admin"
    
    def test_group_command_always_allowed(self):
        """Test commands are always allowed in group chats."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        admin_manager.config = Mock()
        admin_manager.config.allowed_chats = []
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=-100123456,
            is_private=False,
            is_command=True
        )
        
        assert allowed is True
        assert reason is None
    
    def test_group_message_whitelisted_allowed(self):
        """Test regular messages allowed in whitelisted group."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        admin_manager.config = Mock()
        admin_manager.config.allowed_chats = [-100123456]
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=-100123456,
            is_private=False,
            is_command=False
        )
        
        assert allowed is True
        assert reason is None
    
    def test_group_message_not_whitelisted_denied(self):
        """Test regular messages denied in non-whitelisted group."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        admin_manager.config = Mock()
        admin_manager.config.allowed_chats = [-100999999]
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=-100123456,
            is_private=False,
            is_command=False
        )
        
        assert allowed is False
        assert reason == "chat_not_whitelisted"
    
    def test_group_message_empty_whitelist_denied(self):
        """Test regular messages denied when whitelist is empty."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        admin_manager.config = Mock()
        admin_manager.config.allowed_chats = []
        
        service = AccessControlService(admin_manager)
        allowed, reason = service.is_allowed(
            user_id=67890,
            chat_id=-100123456,
            is_private=False,
            is_command=False
        )
        
        assert allowed is False
        assert reason == "chat_not_whitelisted"


class TestCheckAdminAccess:
    """Tests for check_admin_access() method."""
    
    def test_check_admin_access_granted(self):
        """Test admin access check succeeds for admin."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = True
        
        service = AccessControlService(admin_manager)
        has_access, error_msg = service.check_admin_access(12345)
        
        assert has_access is True
        assert error_msg is None
    
    def test_check_admin_access_denied(self):
        """Test admin access check fails for non-admin."""
        admin_manager = Mock()
        admin_manager.is_admin.return_value = False
        
        service = AccessControlService(admin_manager)
        has_access, error_msg = service.check_admin_access(67890)
        
        assert has_access is False
        assert "администратору" in error_msg
    
    def test_check_admin_access_no_manager(self):
        """Test admin access check fails when admin_manager is None."""
        service = AccessControlService(None)
        has_access, error_msg = service.check_admin_access(12345)
        
        assert has_access is False
        assert "недоступна" in error_msg


class TestGetAccessDenialMessage:
    """Tests for get_access_denial_message() method."""
    
    def test_private_non_admin_message(self):
        """Test denial message for private non-admin access."""
        service = AccessControlService(Mock())
        message = service.get_access_denial_message("private_non_admin")
        
        assert "Доступ запрещен" in message
        assert "личных сообщениях" in message
        assert "/admin_set" in message
    
    def test_chat_not_whitelisted_message(self):
        """Test denial message for non-whitelisted chat."""
        service = AccessControlService(Mock())
        message = service.get_access_denial_message("chat_not_whitelisted")
        
        assert "не авторизован" in message
        assert "/admin allowed add" in message
    
    def test_admin_only_message(self):
        """Test denial message for admin-only commands."""
        service = AccessControlService(Mock())
        message = service.get_access_denial_message("admin_only")
        
        assert "администратору" in message
    
    def test_unknown_reason_message(self):
        """Test denial message for unknown reason."""
        service = AccessControlService(Mock())
        message = service.get_access_denial_message("some_unknown_reason")
        
        assert "Доступ запрещен" in message
    
    def test_empty_reason_message(self):
        """Test denial message for empty reason."""
        service = AccessControlService(Mock())
        message = service.get_access_denial_message("")
        
        assert "Доступ запрещен" in message
