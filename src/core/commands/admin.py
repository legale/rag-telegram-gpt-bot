"""Admin command handlers for Telegram bot."""

from __future__ import annotations

from typing import Optional, Union, Dict

from telegram import Update

from src.core.dispatcher import AsyncCommandHandler, CommandContext, CommandResult
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
                error="AdminManager not available",
            )

        update: Optional[Update] = context.metadata.get("update") if context.metadata else None
        if not update or not update.message:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось получить данные сообщения.",
                error="Update object not found in context metadata",
            )

        message = update.message
        text = message.text or ""

        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return CommandResult(
                success=False,
                message=(
                    "Неверный формат команды.\n\n"
                    "Использование: /admin_set <пароль>\n\n"
                    "Пример: /admin_set my_secret_password"
                ),
                error="Missing password argument",
            )

        password = parts[1].strip()

        validation_result = self._validate_password(password)
        if validation_result is not None:
            return validation_result

        return self._set_admin_user(message)

    def _validate_password(self, password: str) -> Optional[CommandResult]:
        """
        Validate admin password.

        Args:
            password: Password to validate

        Returns:
            CommandResult with error if validation failed, None if valid
        """
        if not self.admin_manager.verify_password(password):
            return CommandResult(
                success=False,
                message="Неверный пароль.",
                error="Invalid password",
            )
        return None

    def _set_admin_user(self, message) -> CommandResult:
        """
        Set admin user from message.

        Args:
            message: Telegram message object

        Returns:
            CommandResult with success status and message
        """
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
                data={"user_id": user_id, "username": username},
            )
        except Exception as e:
            syslog2(LOG_ERR, "admin set failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при назначении администратора: {e}",
                error=str(e),
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

    def _get_admin_info(self, user_id: int) -> Optional[CommandResult]:
        """
        Get admin information.

        Args:
            user_id: User ID to check admin status

        Returns:
            CommandResult with error if check failed, None if successful
        """
        if not self.admin_manager:
            return CommandResult(
                success=False,
                message="Система администрирования недоступна.",
                error="AdminManager not available",
            )

        if not self.admin_manager.is_admin(user_id):
            return CommandResult(
                success=False,
                message="Эта команда доступна только администратору.",
                error="User is not admin",
            )

        return None

    def _format_admin_info(self, admin_info: dict) -> str:
        """
        Format admin information for display.

        Args:
            admin_info: Admin information dictionary

        Returns:
            Formatted message string
        """
        return (
            f"**Информация об администраторе:**\n\n"
            f"• Имя: {admin_info.get('first_name', 'N/A')} {admin_info.get('last_name', '')}\n"
            f"• Username: @{admin_info.get('username', 'N/A')}\n"
            f"• User ID: `{admin_info.get('user_id', 'N/A')}`\n"
            f"• Назначен: {admin_info.get('created_at', 'N/A')}"
        )

    async def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle /admin_get command.

        Args:
            context: Command context

        Returns:
            CommandResult with admin information
        """
        user_id_str = context.user_id
        if not user_id_str:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось определить пользователя.",
                error="User ID not found in context",
            )

        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            return CommandResult(
                success=False,
                message="Ошибка: неверный формат ID пользователя.",
                error=f"Invalid user_id: {user_id_str}",
            )

        error_result = self._get_admin_info(user_id)
        if error_result is not None:
            return error_result

        admin_info = self._get_admin_info_data()
        if not admin_info:
            return CommandResult(
                success=True,
                message="Администратор не назначен.",
                data={"admin_set": False},
            )

        return CommandResult(
            success=True,
            message=self._format_admin_info(admin_info),
            data=admin_info,
        )

    def _get_admin_info_data(self) -> Optional[Dict]:
        """
        Get admin information data from admin manager.

        Returns:
            Admin info dictionary or None if no admin is set
        """
        return self.admin_manager.get_admin_info()


class AdminCommandHandler(AsyncCommandHandler):
    """Handler for /admin command (main admin panel)."""

    def __init__(self, admin_router):
        """
        Initialize handler with admin router.

        Args:
            admin_router: AdminCommandRouter instance
        """
        self.admin_router = admin_router

    def _parse_admin_command(self, context: CommandContext) -> Union[tuple[Update, object], CommandResult]:
        """
        Parse admin command from context metadata.

        Args:
            context: Command context

        Returns:
            Tuple of (update, admin_manager) if successful, CommandResult with error otherwise
        """
        update: Optional[Update] = context.metadata.get("update") if context.metadata else None
        admin_manager = context.metadata.get("admin_manager") if context.metadata else None

        if not update:
            return CommandResult(
                success=False,
                message="Ошибка: не удалось получить данные сообщения.",
                error="Update object not found in context metadata",
            )

        if not admin_manager:
            return CommandResult(
                success=False,
                message="Ошибка: AdminManager не доступен.",
                error="AdminManager not found in context metadata",
            )

        return (update, admin_manager)

    async def _execute_admin_command(self, update: Update, admin_manager: object) -> CommandResult:
        """
        Execute admin command using admin router.

        Args:
            update: Telegram Update object
            admin_manager: AdminManager instance

        Returns:
            CommandResult with admin panel response
        """
        try:
            response = await self.admin_router.route(update, None, admin_manager)

            return CommandResult(
                success=True,
                message=response,
                data={"routed": True},
            )
        except Exception as e:
            syslog2(LOG_ERR, "admin command failed", error=str(e))
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении админ-команды: {e}",
                error=str(e),
            )

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
                error="AdminCommandRouter not available",
            )

        parse_result = self._parse_admin_command(context)
        if isinstance(parse_result, CommandResult):
            return parse_result

        update, admin_manager = parse_result
        return await self._execute_admin_command(update, admin_manager)

