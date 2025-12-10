"""
Admin command router for Legale Bot.
Routes /admin commands to appropriate handlers.
"""

from typing import Optional, Callable, Dict, Any, Tuple, List
from telegram import Update
from telegram.ext import ContextTypes
import logging
from src.core.syslog2 import *


class AdminCommandRouter:
    """Routes admin commands to handlers."""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.subcommand_handlers: Dict[str, Dict[str, Callable]] = {}
    
    def register(self, command: str, handler: Callable, subcommand: Optional[str] = None):
        """
        Register a command handler.
        
        Args:
            command: Main command (e.g., 'profile')
            handler: Handler function
            subcommand: Optional subcommand (e.g., 'list')
        """
        if subcommand:
            if command not in self.subcommand_handlers:
                self.subcommand_handlers[command] = {}
            self.subcommand_handlers[command][subcommand] = handler
        else:
            self.handlers[command] = handler
    
    def _check_admin_access(self, admin_manager, user_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check if user has admin access.
        
        Returns:
            Tuple of (is_admin, error_message)
        """
        if not admin_manager or not admin_manager.is_admin(user_id):
            syslog2(LOG_WARNING, "unauthorized admin command", user_id=user_id)
            return False, "Эта команда доступна только администратору."
        return True, None
    
    def _parse_command(self, text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Parse admin command text.
        
        Returns:
            Tuple of (command, subcommand, args)
        """
        parts = text.split()
        
        if len(parts) == 1:
            return None, None, []
        
        command = parts[1]
        
        if len(parts) > 2:
            subcommand = parts[2]
            args = parts[3:]
            return command, subcommand, args
        
        return command, None, []
    
    async def _execute_handler(self, handler: Callable, update: Update, 
                               context: ContextTypes.DEFAULT_TYPE, 
                               admin_manager, args: List[str], 
                               command_name: str) -> str:
        """
        Execute a command handler with error handling.
        
        Args:
            handler: Handler function to execute
            update: Telegram update
            context: Bot context
            admin_manager: AdminManager instance
            args: Command arguments
            command_name: Name of command for error logging
        
        Returns:
            Response message
        """
        try:
            return await handler(update, context, admin_manager, args)
        except Exception as e:
            syslog2(LOG_ERR, "handler failed", command=command_name, error=str(e))
            return f"Ошибка при выполнении команды: {e}"
    
    async def _route_with_subcommand(self, command: str, subcommand: str, 
                                     args: List[str], update: Update,
                                     context: ContextTypes.DEFAULT_TYPE,
                                     admin_manager) -> str:
        """Route command with subcommand."""
        # Try to find subcommand handler
        if command in self.subcommand_handlers and subcommand in self.subcommand_handlers[command]:
            handler = self.subcommand_handlers[command][subcommand]
            return await self._execute_handler(
                handler, update, context, admin_manager, args, f"{command}/{subcommand}"
            )
        
        # Fallback: try direct handler with all args
        if command in self.handlers:
            handler = self.handlers[command]
            full_args = [subcommand] + args
            return await self._execute_handler(
                handler, update, context, admin_manager, full_args, command
            )
        
        # Command exists but subcommand is invalid
        if command in self.subcommand_handlers:
            return f"Неизвестная подкоманда: {subcommand}\n\nИспользуйте /admin help {command}"
        
        return f"Команда '{command}' не найдена."
    
    async def _route_without_subcommand(self, command: str, update: Update,
                                        context: ContextTypes.DEFAULT_TYPE,
                                        admin_manager) -> str:
        """Route command without subcommand."""
        # Try direct handler
        if command in self.handlers:
            handler = self.handlers[command]
            return await self._execute_handler(
                handler, update, context, admin_manager, [], command
            )
        
        # Command requires subcommand
        if command in self.subcommand_handlers:
            subcommands = ", ".join(self.subcommand_handlers[command].keys())
            return f"Команда '{command}' требует подкоманду.\n\nДоступные: {subcommands}"
        
        return f"Неизвестная команда: {command}\n\nИспользуйте /admin help"
    
    async def route(self, update: Update, context: ContextTypes.DEFAULT_TYPE, admin_manager) -> str:
        """
        Route admin command to appropriate handler.
        
        Args:
            update: Telegram update
            context: Bot context
            admin_manager: AdminManager instance
        
        Returns:
            Response message
        """
        message = update.message
        user_id = message.from_user.id
        
        # Check admin access
        is_admin, error = self._check_admin_access(admin_manager, user_id)
        if not is_admin:
            return error
        
        # Parse command
        command, subcommand, args = self._parse_command(message.text)
        
        # No command - show main menu
        if command is None:
            return self._main_menu()
        
        # Route based on presence of subcommand
        if subcommand:
            return await self._route_with_subcommand(
                command, subcommand, args, update, context, admin_manager
            )
        else:
            return await self._route_without_subcommand(
                command, update, context, admin_manager
            )
    
    def _main_menu(self) -> str:
        """Generate main admin menu."""
        return (
            "**Панель администратора**\n\n"
            "**Управление профилями:**\n"
            "• `/admin profile list` - список профилей\n"
            "• `/admin profile create <name>` - создать профиль\n"
            "• `/admin profile get` - показать текущий профиль\n"
            "• `/admin profile set <name>` - переключить профиль\n"
            "• `/admin profile delete <name>` - удалить профиль\n"
            "• `/admin profile info [name]` - информация о профиле\n\n"
            "**Загрузка данных:**\n"
            "• `/admin ingest` - загрузить данные (отправьте JSON файл)\n"
            "• `/admin ingest clear` - очистить данные профиля\n"
            "• `/admin ingest status` - статус загрузки\n\n"
            "**Доступ:**\n"
            "• `/admin allowed list` - список разрешенных чатов\n"
            "• `/admin allowed lookup <chat_name_substring>` - поиск чатов по имени (подстрока)\n"
            "• `/admin allowed add <id>` - разрешить чат\n"
            "• `/admin allowed remove <id>` - запретить чат\n\n"
            "**Статистика и мониторинг:**\n"
            "• `/admin stats` - общая статистика\n"
            "• `/admin health` - проверка здоровья системы\n"
            "• `/admin logs [lines]` - просмотр логов\n\n"
            "**Управление:**\n"
            "• `/admin restart` - перезапустить бота\n"
            "• `/admin reload` - перезагрузить конфигурацию\n\n"
            "**Справка:**\n"
            "• `/admin help` - полная справка\n"
            "• `/admin help <command>` - справка по команде\n"
        )
