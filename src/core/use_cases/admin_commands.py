"""Admin command handlers for Telegram bot."""

from __future__ import annotations

from typing import Optional
from telegram import Update

from ..dispatcher import AsyncCommandHandler, CommandContext, CommandResult
from src.lib.syslog2 import *


class AdminSetCommandHandler(AsyncCommandHandler):
    """Handler for /admin_set command."""

    def __init__(self, admin_manager):
        """
        Initialize handler with admin manager.

        Args:
            admin_manager: AdminManager instance
        """
        self.admin_manager = admin_manager

    async def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle /admin_set command.

        Args:
            context: Command context with args containing password

        Returns:
            CommandResult with success status and message
        """
        if not self.admin_manager:
            return CommandResult(
                success=False,
                message="Система администрирования недоступна. Установите ADMIN_PASSWORD в .env файле.",
                error="AdminManager not available"
            )

        # Get Update object from metadata
        update: Optional[Update] = context.metadata.get("update") if context.metadata else None
        if not update or not update.message:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось получить данные сообщения.",
                error="Update object not found in context metadata"
            )

        message = update.message
        text = message.text or ""

        # Parse arguments
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(
                success=False,
                message=(
                    "Неверный формат команды.\n\n"
                    "Использование: /admin_set <пароль>\n\n"
                    "Пример: /admin_set my_secret_password"
                ),
                error="Missing password argument"
            )

        password = parts[1].strip()

        if not self.admin_manager.verify_password(password):
            return CommandResult(
                success=False,
                message="Неверный пароль.",
                error="Invalid password"
            )

        # Set admin
        user = message.from_user
        user_id = user.id
        username = user.username or "unknown"
        first_name = user.first_name or "Unknown"
        last_name = user.last_name

        try:
            self.admin_manager.set_admin(user_id, username, first_name, last_name)
            full_name = f"{first_name} {last_name}".strip() if last_name else first_name
            syslog2(LOG_NOTICE, "admin set", full_name=full_name, user_id=user_id)

            return CommandResult(
                success=True,
                message=(
                    f"✅ Вы назначены администратором!\n\n"
                    f"**Информация:**\n"
                    f"• Имя: {full_name}\n"
                    f"• Username: @{username}\n"
                    f"• User ID: `{user_id}`\n\n"
                    f"Теперь вам доступны команды `/admin`."
                ),
                data={"user_id": user_id, "username": username}
            )
        except Exception as e:
            syslog2(LOG_ERR, "admin set failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при назначении администратора: {e}",
                error=str(e)
            )


class AdminGetCommandHandler(AsyncCommandHandler):
    """Handler for /admin_get command."""

    def __init__(self, admin_manager):
        """
        Initialize handler with admin manager.

        Args:
            admin_manager: AdminManager instance
        """
        self.admin_manager = admin_manager

    async def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle /admin_get command.

        Args:
            context: Command context

        Returns:
            CommandResult with admin information
        """
        if not self.admin_manager:
            return CommandResult(
                success=False,
                message="Система администрирования недоступна.",
                error="AdminManager not available"
            )

        # Get user_id from context
        user_id_str = context.user_id
        if not user_id_str:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось определить пользователя.",
                error="User ID not found in context"
            )

        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            return CommandResult(
                success=False,
                message="Ошибка: неверный формат ID пользователя.",
                error=f"Invalid user_id: {user_id_str}"
            )

        if not self.admin_manager.is_admin(user_id):
            return CommandResult(
                success=False,
                message="Эта команда доступна только администратору.",
                error="User is not admin"
            )

        # Get admin info
        admin_info = self.admin_manager.get_admin_info()
        if not admin_info:
            return CommandResult(
                success=True,
                message="Администратор не назначен.",
                data={"admin_set": False}
            )

        return CommandResult(
            success=True,
            message=(
                f"**Информация об администраторе:**\n\n"
                f"• Имя: {admin_info.get('first_name', 'N/A')} {admin_info.get('last_name', '')}\n"
                f"• Username: @{admin_info.get('username', 'N/A')}\n"
                f"• User ID: `{admin_info.get('user_id', 'N/A')}`\n"
                f"• Назначен: {admin_info.get('created_at', 'N/A')}"
            ),
            data=admin_info
        )


class AdminCommandHandler(AsyncCommandHandler):
    """Handler for /admin command (main admin panel)."""

    def __init__(self, admin_router):
        """
        Initialize handler with admin router.

        Args:
            admin_router: AdminCommandRouter instance
        """
        self.admin_router = admin_router

    async def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle /admin command.

        Args:
            context: Command context with admin subcommand and args

        Returns:
            CommandResult with admin panel response
        """
        if not self.admin_router:
            return CommandResult(
                success=False,
                message="Админ-панель недоступна. Проверьте конфигурацию бота.",
                error="AdminCommandRouter not available"
            )

        # Get Update object and admin_manager from metadata
        update: Optional[Update] = context.metadata.get("update") if context.metadata else None
        admin_manager = context.metadata.get("admin_manager") if context.metadata else None

        if not update:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось получить данные сообщения.",
                error="Update object not found in context metadata"
            )

        if not admin_manager:
            return CommandResult(
                success=False,
                message="Ошибка: AdminManager не доступен.",
                error="AdminManager not found in context metadata"
            )

        try:
            # Delegate to AdminCommandRouter
            # AdminCommandRouter expects the full command text
            message_text = update.message.text if update.message else "/admin"
            response = await self.admin_router.route(update, None, admin_manager)
            
            return CommandResult(
                success=True,
                message=response,
                data={"routed": True}
            )
        except Exception as e:
            syslog2(LOG_ERR, "admin command failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении админ-команды: {e}",
                error=str(e)
            )

