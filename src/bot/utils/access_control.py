"""
Access control utilities for Legale Bot.

Provides centralized access control logic including:
- Admin access checks
- Chat whitelist checks
- Private vs group chat logic
- Access denial messages
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AccessControlService:
    """Service for managing access control logic."""
    
    def __init__(self, admin_manager):
        """
        Initialize AccessControlService.
        
        Args:
            admin_manager: AdminManager instance
        """
        self.admin_manager = admin_manager
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is an administrator.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is admin, False otherwise
        """
        if not self.admin_manager:
            return False
        return self.admin_manager.is_admin(user_id)
    
    def is_allowed(self, user_id: int, chat_id: int, 
                   is_private: bool, is_command: bool) -> tuple[bool, Optional[str]]:
        """
        Check if user/chat is allowed to interact with the bot.
        
        Logic:
        - Private chats: Only admins are allowed
        - Group chats: Commands are always allowed, messages only if chat is whitelisted
        - Admins: Always allowed everywhere
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            is_private: True if private chat
            is_command: True if message is a command
            
        Returns:
            Tuple of (is_allowed, denial_reason)
        """
        # Admins are always allowed
        if self.is_admin(user_id):
            logger.debug(f"Access granted: user {user_id} is admin")
            return True, None
        
        if is_private:
            # Private messages: only admins allowed
            logger.debug(f"Access denied: private message from non-admin {user_id}")
            return False, "private_non_admin"
        else:
            # Group/supergroup/channel
            # Commands are always allowed
            if is_command:
                logger.debug(f"Access granted: command in chat {chat_id}")
                return True, None
            
            # Regular messages: check whitelist
            config = self.admin_manager.config
            if chat_id in config.allowed_chats:
                logger.debug(f"Access granted: chat {chat_id} is whitelisted")
                return True, None
            else:
                logger.debug(f"Access denied: chat {chat_id} not whitelisted")
                return False, "chat_not_whitelisted"
    
    def check_admin_access(self, user_id: int) -> tuple[bool, Optional[str]]:
        """
        Check if user has admin access.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Tuple of (has_access, error_message)
        """
        if not self.admin_manager:
            return False, "❌ Система администрирования недоступна."
        
        if not self.is_admin(user_id):
            logger.warning(f"Unauthorized admin command attempt from user {user_id}")
            return False, "❌ Эта команда доступна только администратору."
        
        return True, None
    
    def get_access_denial_message(self, reason: str) -> str:
        """
        Get a user-friendly access denial message.
        
        Args:
            reason: Denial reason code
            
        Returns:
            Formatted denial message
        """
        messages = {
            "private_non_admin": (
                "❌ Доступ запрещен.\n\n"
                "В личных сообщениях бот доступен только администратору.\n"
                "Используйте команду /admin_set для назначения администратора."
            ),
            "chat_not_whitelisted": (
                "❌ Этот чат не авторизован.\n\n"
                "Администратор должен добавить чат в белый список:\n"
                "`/admin allowed add <chat_id>`"
            ),
            "admin_only": "❌ Эта команда доступна только администратору.",
            "unknown": "❌ Доступ запрещен.",
        }
        
        return messages.get(reason, messages["unknown"])
