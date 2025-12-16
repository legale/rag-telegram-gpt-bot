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
from src.lib.syslog2 import *
from src.core.access_control import check_access as core_check_access


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
                   is_private: bool, is_command: bool, 
                   command_text: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Check if user/chat is allowed to interact with the bot.
        
        Logic:
        - Private chats: Only admins are allowed, except /admin_set command
        - Group chats: Commands are always allowed, messages only if chat is whitelisted
        - Admins: Always allowed everywhere
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            is_private: True if private chat
            is_command: True if message is a command
            command_text: Optional command text (e.g., "/admin_set password")
            
        Returns:
            Tuple of (is_allowed, denial_reason)
        """
        config = getattr(self.admin_manager, "config", None)
        raw_allowed_chats = getattr(config, "allowed_chats", [])

        allowed_chats: list[int]
        if isinstance(raw_allowed_chats, list):
            allowed_chats = raw_allowed_chats
        elif isinstance(raw_allowed_chats, (tuple, set)):
            allowed_chats = list(raw_allowed_chats)
        else:
            try:
                allowed_chats = list(raw_allowed_chats)
            except TypeError:
                allowed_chats = []

        allowed, reason = core_check_access(
            user_id=user_id,
            chat_id=chat_id,
            is_private=is_private,
            is_command=is_command,
            allowed_chats=allowed_chats,
            is_admin=self.is_admin,
            command_text=command_text,
        )

        if allowed:
            if reason is None:
                syslog2(LOG_DEBUG, "access granted", user_id=user_id, chat_id=chat_id, is_private=is_private, is_command=is_command)
            return True, None

        syslog2(LOG_DEBUG, "access denied", user_id=user_id, chat_id=chat_id, reason=reason)
        return False, reason
    
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
