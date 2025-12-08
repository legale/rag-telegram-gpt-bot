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
from src.core.syslog2 import *


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
            syslog2(LOG_DEBUG, "access granted admin", user_id=user_id)
            return True, None
        
        if is_private:
            # Private messages: only admins allowed
            syslog2(LOG_DEBUG, "access denied private", user_id=user_id)
            return False, "private_non_admin"
        else:
            # Group/supergroup/channel
            # Commands are always allowed
            if is_command:
                syslog2(LOG_DEBUG, "access granted command", chat_id=chat_id)
                return True, None
            
            # Regular messages: check whitelist
            config = self.admin_manager.config
            if chat_id in config.allowed_chats:
                syslog2(LOG_DEBUG, "access granted whitelist", chat_id=chat_id)
                return True, None
            else:
                syslog2(LOG_DEBUG, "access denied not whitelisted", chat_id=chat_id)
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
            return False, "Система администрирования недоступна."
        
        if not self.is_admin(user_id):
            syslog2(LOG_WARNING, "unauthorized admin command", user_id=user_id)
            return False, "Эта команда доступна только администратору."
        
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
                "Доступ запрещен.\n\n"
                "В личных сообщениях бот доступен только администратору.\n"
                "Используйте команду /admin_set для назначения администратора."
            ),
            "chat_not_whitelisted": (
                "Этот чат не авторизован.\n\n"
                "Администратор должен добавить чат в белый список:\n"
                "`/admin allowed add <chat_id>`"
            ),
            "admin_only": "Эта команда доступна только администратору.",
            "unknown": "Доступ запрещен.",
        }
        
        return messages.get(reason, messages["unknown"])
