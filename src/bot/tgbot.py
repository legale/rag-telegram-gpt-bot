#!/usr/bin/env python3
"""
Telegram Bot Webhook Daemon for Legale Bot.

This module provides:
- FastAPI webhook endpoint for Telegram updates
- CLI utilities for webhook registration/deletion
- Daemon and foreground modes with log level control
- Persistent in-memory LegaleBot instance
"""

import sys
import os
import logging
import signal
import inspect
from types import SimpleNamespace
from typing import Optional, Dict, List, Tuple, Union, Callable
from contextlib import asynccontextmanager

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application
import uvicorn
from dotenv import load_dotenv

from src.bot.core import LegaleBot
from src.bot.admin import AdminManager
from src.bot.admin_router import AdminCommandRouter
from src.core.message_search import search_message_contents
from src.bot.admin_commands import ProfileCommands, HelpCommands, IngestCommands, StatsCommands, ControlCommands, SettingsCommands, ModelCommands, SystemPromptCommands
from src.bot.admin_tasks import TaskManager
from src.bot.utils import AccessControlService, FrequencyController, ErrorHandler
from src.bot.utils.response_formatter import ResponseFormatter
from src.bot.command_parser import parse_find_command_args as parse_find_args_common
from src.lib.syslog2 import *
from src.app.types import AppRequest, AppResponse

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger("legale_tgbot")


class RuntimeContext:
    """Runtime context for storing bot state and dependencies."""
    
    def __init__(self):
        self.bot_instance: Optional[LegaleBot] = None
        self.telegram_app: Optional[Application] = None
        self.admin_manager: Optional[AdminManager] = None
        self.admin_router: Optional[AdminCommandRouter] = None
        self.profile_manager = None  # Will be initialized in lifespan
        self.task_manager: Optional[TaskManager] = None
        self.ingest_commands: Optional[IngestCommands] = None
        self.command_dispatcher = None  # CommandDispatcher instance
        self.access_control: Optional[AccessControlService] = None
        self.frequency_controller: FrequencyController = FrequencyController()
        self.debug_rag_mode: bool = False


# Global runtime context (replaces individual global variables)
_runtime_context: Optional[RuntimeContext] = None


def get_runtime_context() -> RuntimeContext:
    """Get the global runtime context."""
    global _runtime_context
    if _runtime_context is None:
        _runtime_context = RuntimeContext()
    return _runtime_context


# Backward compatibility: expose global variables as properties
# These will be deprecated but kept for compatibility during transition
def _get_bot_instance() -> Optional[LegaleBot]:
    return get_runtime_context().bot_instance


def _get_admin_manager() -> Optional[AdminManager]:
    return get_runtime_context().admin_manager


def _get_admin_router() -> Optional[AdminCommandRouter]:
    return get_runtime_context().admin_router


def _get_task_manager() -> Optional[TaskManager]:
    return get_runtime_context().task_manager


def _get_ingest_commands() -> Optional[IngestCommands]:
    return get_runtime_context().ingest_commands


def _get_command_dispatcher():
    return get_runtime_context().command_dispatcher


def _get_access_control() -> Optional[AccessControlService]:
    return get_runtime_context().access_control


def _get_frequency_controller() -> FrequencyController:
    return get_runtime_context().frequency_controller


def _get_debug_rag_mode() -> bool:
    return get_runtime_context().debug_rag_mode


def _set_debug_rag_mode(value: bool) -> None:
    get_runtime_context().debug_rag_mode = value


class MessageHandler:
    """Handles message routing and command processing."""
    
    def __init__(self, bot_instance, admin_manager, admin_router, ctx: Optional[RuntimeContext] = None):
        self.bot = bot_instance
        self.admin_manager = admin_manager
        self.admin_router = admin_router
        if ctx is None:
            ctx = RuntimeContext()
            ctx.bot_instance = bot_instance
            ctx.admin_manager = admin_manager
            ctx.admin_router = admin_router
        self.ctx = ctx
    
    async def handle_start_command(self) -> str:
        """Handle /start command."""
        return (
            "Привет!\n\n"
            "Используйте /help для справки."
        )
    
    async def handle_help_command(self) -> str:
        """Handle /help command."""
        return (
            "Я анализирую историю чата и отвечаю на вопросы.\n\n"
            "Доступные команды:\n"
            "• /start — приветствие\n"
            "• /help — эта справка\n"
            "• /reset — сбросить контекст разговора\n"
            "• /tokens — показать использование токенов\n"
            "• /model — переключить модель LLM\n"
            "• /find <запрос> — поиск сообщений по запросу\n"
            "• /admin_set <пароль> — назначить себя администратором\n"
            "• /admin_get — показать информацию об администраторе (только для админа)\n"
            "• /admin — панель администратора (только для админа)\n"
            "• /id — показать ID текущего чата\n\n"
            "Просто напишите свой вопрос!"
        )
    
    async def handle_reset_command(self) -> str:
        """Handle /reset command."""
        try:
            return self.bot.reset_context()
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "сбросе контекста", user_message="Ошибка при сбросе контекста.")
    
    async def handle_tokens_command(self) -> str:
        """Handle /tokens command."""
        try:
            usage = self.bot.get_token_usage()
            response = (
                f"Использование токенов:\n\n"
                f"Текущее: {usage['current_tokens']:,}\n"
                f"Максимум: {usage['max_tokens']:,}\n"
                f"Использовано: {usage['percentage']}%\n\n"
            )
            if usage["percentage"] > 80:
                response += "Приближаетесь к лимиту! Используйте /reset для сброса."
            elif usage["percentage"] > 50:
                response += "Контекст заполнен наполовину."
            else:
                response += "Достаточно места для разговора."
            return response
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "получении информации о токенах", user_message="Ошибка при получении информации о токенах.")
    
    async def handle_model_command(self) -> str:
        """Handle /model command."""
        try:
            msg = self.bot.get_model()
            # Save new model to config
            if self.admin_manager:
                self.admin_manager.config.current_model = self.bot.current_model_name
            return msg
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "переключении модели", user_message="Ошибка при переключении модели.")
    
    async def handle_admin_set_command(self, text: str, message) -> str:
        """Handle /admin_set command."""
        if not self.admin_manager:
            return "Система администрирования недоступна. Установите ADMIN_PASSWORD в .env файле."
        
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return (
                "Неверный формат команды.\n\n"
                "Использование: /admin_set <пароль>\n\n"
                "Пример: /admin_set my_secret_password"
            )
        
        password = parts[1].strip()
        
        if self.admin_manager.verify_password(password):
            user = message.from_user
            user_id = user.id
            username = user.username or "unknown"
            first_name = user.first_name or "Unknown"
            last_name = user.last_name
            
            try:
                self.admin_manager.set_admin(user_id, username, first_name, last_name)
                full_name = f"{first_name} {last_name}".strip() if last_name else first_name
                syslog2(LOG_NOTICE, "admin set", full_name=full_name, user_id=user_id)
                return (
                    f"Вы успешно назначены администратором!\n\n"
                    f"Имя: {full_name}\n"
                    f"🆔 ID: {user_id}\n"
                    f"Username: @{username}"
                )
            except Exception as e:
                return ErrorHandler.handle_error_static(e, "назначении администратора", user_message="Ошибка при назначении администратора.")
        else:
            syslog2(LOG_WARNING, "failed admin set attempt", user_id=message.from_user.id)
            return "Неверный пароль."
    
    async def handle_admin_get_command(self, user_id: int) -> str:
        """Handle /admin_get command."""
        if not self.admin_manager:
            return "Система администрирования недоступна."
        
        if not self.admin_manager.is_admin(user_id):
            syslog2(LOG_WARNING, "unauthorized admin get attempt", user_id=user_id)
            return "Эта команда доступна только администратору."
        
        admin_info = self.admin_manager.get_admin()
        if admin_info:
            return (
                f"Администратор бота:\n\n"
                f"Имя: {admin_info['full_name']}\n"
                f"ID: {admin_info['user_id']}\n"
                f"Username: @{admin_info['username']}"
            )
        else:
            return "Администратор не назначен."
    
    async def handle_admin_command(self, update: Update) -> str:
        """Handle /admin command."""
        if not self.admin_router:
            return "Админ-панель недоступна. Проверьте конфигурацию бота."
        
        try:
            return await self.admin_router.route(update, None, self.admin_manager)
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "выполнении админ-команды", user_message=f"Ошибка при выполнении админ-команды: {e}")
    
    def _parse_find_command_args(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse find command arguments (threshold and query).
        
        Args:
            text: Command text (e.g., "/find 2.0 vpn туннель" or "/find vpn туннель")
            
        Returns:
            Tuple of (threshold, query) or (None, error_message)
            threshold uses config default if not specified
        """
        return parse_find_args_common(text, admin_manager=self.admin_manager)
    
    async def _send_find_results(self, update: Update, search_query: str, threshold: float) -> Optional[str]:
        """
        Execute simple search in embeddings and send results filtered by cosine distance threshold.
        
        Args:
            update: Telegram update object
            search_query: Search query string
            threshold: Cosine distance threshold (filter results with distance <= threshold)
            
        Returns:
            Error message if failed, None or empty string if successful
        """
        try:
            # get db from bot instance
            db = self.bot.db
            
            # Get debug_rag from runtime context
            debug_rag_mode = _get_debug_rag_mode()
            
            # Get profile paths for creating HybridSearch
            paths = _get_profile_paths(self.ctx)
            profile_dir = str(paths['profile_dir'])
            
            # Create HybridSearch use case via bootstrap
            from src.app.bootstrap import create_hybrid_search
            hybrid_search = create_hybrid_search(
                db_url=paths["db_url"],
                vector_db_path=str(paths["vector_db_path"]),
                embedding_client=self.bot.embedding_client,
                profile_dir=profile_dir
            )
            
            # Convert threshold from distance to similarity score (threshold is distance, score = 1 - distance)
            similarity_threshold = 1.0 - threshold if threshold <= 1.0 else 0.0
            
            # Perform search using HybridSearch
            search_results = hybrid_search.search(
                query=search_query,
                top_k=100,
                threshold=similarity_threshold,
                enrich_with_messages=False  # We'll get messages from DB separately
            )
            
            if not search_results:
                return f'по запросу "{search_query}" ничего не найдено (distance <= {threshold})'
            
            # Convert SearchResult objects to dict format
            from src.core.message_search import convert_search_results_to_dict, _prepare_message_parts_from_results
            filtered_results = convert_search_results_to_dict(search_results)
            
            # Prepare message parts from filtered results
            message_parts_list = _prepare_message_parts_from_results(db, filtered_results, debug_rag_mode)
            
            if not message_parts_list:
                return f'по запросу "{search_query}" ничего не найдено'
            
            # send each message part as separate message
            chat_id = update.message.chat_id
            total_parts = await _send_message_parts_unified(
                chat_id=chat_id,
                message_parts_list=message_parts_list,
                log_context={
                    "query": search_query,
                    "threshold": threshold,
                    "chunks_found": len(filtered_results),
                }
            )
            
            syslog2(
                LOG_ALERT,
                "find command response sent",
                chat_id=chat_id,
                query=search_query,
                threshold=threshold,
                chunks_found=len(filtered_results),
                messages=len(message_parts_list),
                parts=total_parts,
            )
            # return empty string to signal "handled, but ничего не слать отдельно"
            return ""
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "выполнении поиска", user_message=f"Ошибка при выполнении поиска: {e}")
    
    async def handle_find_command(self, text: str, update: Update) -> Optional[str]:
        """Handle /find command."""
        syslog2(LOG_ALERT, "handle_find_command", text=text)
        
        # Parse arguments
        threshold, result = self._parse_find_command_args(text)
        if threshold is None:
            return result  # result is error message
        
        # Send results
        return await self._send_find_results(update, result, threshold)

    
    def _prepare_system_prompt(self) -> str:
        """
        Get system prompt from config.
        
        Returns:
            System prompt template string
        """
        return self.admin_manager.config.get_system_prompt()
    
    def _print_rag_debug_info(self, text: str, n_results: int = 3) -> None:
        """
        Print RAG debug information (chunks, prompts, token count).
        
        Args:
            text: User query text
            n_results: Number of chunks to show
        """
        debug_rag_mode = _get_debug_rag_mode()
        if not debug_rag_mode:
            return
        
        debug_info = self.bot.get_rag_debug_info(text, n_results=n_results)
        print("\n" + "=" * 70)
        print("RAG DEBUG INFO")
        print("=" * 70)
        print(f"\nRetrieved Chunks: {len(debug_info['chunks'])}")
        for i, chunk in enumerate(debug_info['chunks'], 1):
            print(f"\n--- Chunk {i} (score: {chunk.get('score', 'N/A'):.3f}, source: {chunk.get('source', 'unknown')}) ---")
            meta = chunk.get('metadata', {})
            if meta.get('topic_l2_title'):
                print(f"Category: {meta['topic_l2_title']}")
            if meta.get('topic_l1_title'):
                print(f"Topic: {meta['topic_l1_title']}")
            print(f"Text preview: {chunk['text'][:200]}...")
            if len(chunk['text']) > 200:
                print(f"  (full length: {len(chunk['text'])} chars)")
        print("\n" + "-" * 70)
        print(f"System Prompt ({len(debug_info['prompt'])} chars):")
        print("-" * 70)
        print(debug_info['prompt'])
        print("-" * 70)
        print(f"\nUser Prompt ({len(text)} chars):")
        print("-" * 70)
        print(text)
        print("-" * 70)
        print(f"\nToken count: {debug_info.get('token_count', 'N/A')}")
        print("=" * 70 + "\n")
    
    async def handle_user_query(self, text: str, respond: bool) -> str:
        """Handle regular user query to bot."""
        syslog2(LOG_ALERT, "handle_user_query", text=text)
        try:
            # Prepare system prompt
            system_prompt_template = self._prepare_system_prompt()
            
            # Debug RAG mode - show retrieved chunks and prompts
            debug_rag_mode = _get_debug_rag_mode()
            if debug_rag_mode and respond:
                self._print_rag_debug_info(text, n_results=3)
            
            return self.bot.chat(text, respond=respond, system_prompt_template=system_prompt_template)
        except Exception as e:
            return ErrorHandler.handle_error_static(e, "обработке вашего запроса", user_message=f"Произошла ошибка при обработке вашего запроса. error={e}")
    
    def _get_command_and_args(self, text: str) -> Tuple[Optional[str], str]:
        """
        Extract command and arguments from text.
        
        Args:
            text: Command text
            
        Returns:
            Tuple of (command, remaining_text)
        """
        from src.app.main_cli import parse_command
        return parse_command(text)
    
    async def route_command(self, text: str, update: Update) -> Optional[str]:
        """Route command to appropriate handler using CommandDispatcher."""
        syslog2(LOG_ALERT, "route_command", text=text)
        
        # Use CommandDispatcher if available
        command_dispatcher = _get_command_dispatcher()
        if command_dispatcher:
            from src.app.main_cli import handle_command_async
            message = update.message
            user_id = str(message.from_user.id) if message.from_user else None
            chat_id = str(message.chat_id) if message.chat_id else None
            
            # Create metadata for context
            metadata = {
                "update": update,
                "message": message,
                "admin_manager": self.admin_manager,
                "admin_router": self.admin_router
            }
            
            # Handle command using unified async handler
            result_message, result_data = await handle_command_async(
                command=text,
                dispatcher=command_dispatcher,
                user_id=user_id,
                chat_id=chat_id,
                metadata=metadata
            )
            
            if result_message is not None:
                # Check if result needs special formatting (find command)
                if result_data and result_data.get("needs_formatting"):
                    # Send formatted search results
                    message_parts_list = result_data.get("message_parts_list", [])
                    query = result_data.get("query", "")
                    rag_method = result_data.get("rag_method", "hybrid")
                    chat_id = update.message.chat_id
                    
                    total_parts = await _send_message_parts_unified(
                        chat_id=chat_id,
                        message_parts_list=message_parts_list,
                        empty_message=f'по запросу "{query}" ничего не найдено (метод: {rag_method})',
                        log_context={
                            "query": query,
                            "rag_method": rag_method,
                            "results_count": len(message_parts_list),
                        }
                    )
                    
                    syslog2(
                        LOG_ALERT,
                        "find command response sent",
                        chat_id=chat_id,
                        query=query,
                        rag_method=rag_method,
                        results_count=len(message_parts_list),
                        parts=total_parts,
                    )
                    return ""  # Empty string to signal "handled, but ничего не слать отдельно"
                
                return result_message
        
        return None  # Not a recognized command


def _register_command_group(
    router: AdminCommandRouter,
    group_name: str,
    command_instance: object,
    methods: Dict[str, str]
) -> None:
    """
    Register a group of commands for an admin router.
    
    Args:
        router: AdminCommandRouter instance
        group_name: Command group name (e.g., "profile", "ingest")
        command_instance: Instance of command class (e.g., ProfileCommands)
        methods: Dictionary mapping method names to subcommand names.
                 If value is None or empty string, register as main command.
                 Example: {"list_profiles": "list", "create_profile": "create", "show_stats": None}
    """
    for method_name, subcommand in methods.items():
        method = getattr(command_instance, method_name)
        if subcommand:
            router.register(group_name, method, subcommand)
        else:
            router.register(group_name, method)


def _get_profile_paths(ctx: Optional[RuntimeContext] = None) -> Dict:
    """
    Get profile paths for current active profile.
    
    Args:
        ctx: RuntimeContext instance
        
    Returns:
        Dictionary with profile paths
        
    Raises:
        RuntimeError: If profile_manager is not initialized
    """
    if ctx is None:
        ctx = get_runtime_context()

    if ctx.profile_manager is None:
        raise RuntimeError("profile_manager is not initialized")
    
    return ctx.profile_manager.get_profile_paths()


def _create_admin_manager(profile_dir: str) -> AdminManager:
    """
    Create and initialize AdminManager for profile.
    
    Args:
        profile_dir: Profile directory path
        
    Returns:
        Initialized AdminManager instance
    """
    admin_manager_local = AdminManager(profile_dir)
    syslog2(LOG_NOTICE, "admin manager initialized", profile_dir=str(profile_dir))
    return admin_manager_local


def _map_log_level_to_constant(log_level: Union[str, int]) -> int:
    """
    Map log level string or number to syslog2 constant.
    
    Args:
        log_level: Log level as string (e.g., "INFO", "LOG_DEBUG") or int constant
        
    Returns:
        syslog2 log level constant (LOG_ALERT, LOG_CRIT, LOG_ERR, etc.)
    """
    if isinstance(log_level, int):
        return log_level
    
    log_level_upper = str(log_level).upper()
    log_level_map = {
        "LOG_ALERT": LOG_ALERT,
        "LOG_CRIT": LOG_CRIT,
        "LOG_ERR": LOG_ERR,
        "LOG_WARNING": LOG_WARNING,
        "LOG_NOTICE": LOG_NOTICE,
        "LOG_INFO": LOG_INFO,
        "LOG_DEBUG": LOG_DEBUG,
        "ALERT": LOG_ALERT,
        "CRIT": LOG_CRIT,
        "ERR": LOG_ERR,
        "WARNING": LOG_WARNING,
        "NOTICE": LOG_NOTICE,
        "INFO": LOG_INFO,
        "DEBUG": LOG_DEBUG,
    }
    return log_level_map.get(log_level_upper, LOG_WARNING)


def _get_bot_configuration(admin_manager_local: AdminManager, args: Optional[SimpleNamespace]) -> Tuple[str, bool, int, str]:
    """
    Extract bot configuration from admin manager and args.
    
    Args:
        admin_manager_local: AdminManager instance
        args: Optional namespace with configuration overrides
        
    Returns:
        Tuple of (model_name, debug_rag, log_level, retrieval_type)
    """
    model_name = admin_manager_local.config.current_model or "openai/gpt-oss-20b:free"
    debug_rag = getattr(args, 'debug_rag', False) if args else False
    
    # Получаем log_level из args, преобразуем строку в константу если нужно
    log_level = getattr(args, 'log_level', LOG_WARNING) if args else LOG_WARNING
    log_level = _map_log_level_to_constant(log_level)
    
    retrieval_type = getattr(args, 'retrieval_type', 'hybrid') if args else 'hybrid'
    
    return model_name, debug_rag, log_level, retrieval_type


def _create_legale_bot(
    paths: Dict,
    model_name: str,
    log_level: int,
    debug_rag: bool,
    profile_dir: str,
    retrieval_type: str = "hybrid",
    ctx: Optional[RuntimeContext] = None,
) -> LegaleBot:
    """
    Create and initialize LegaleBot instance.
    
    Args:
        paths: Profile paths dictionary
        model_name: Model name to use
        log_level: Logging level
        debug_rag: Debug RAG flag
        profile_dir: Profile directory path
        ctx: RuntimeContext instance
        retrieval_type: Retrieval type ("hybrid", "fts_only", "vector_only")
        
    Returns:
        Initialized LegaleBot instance
    """
    if ctx is None:
        ctx = get_runtime_context()

    bot_instance_local = LegaleBot(
        db_url=paths["db_url"],
        vector_db_path=str(paths["vector_db_path"]),
        model_name=model_name,
        log_level=log_level,
        debug_rag=debug_rag,
        profile_dir=profile_dir,
        retrieval_type=retrieval_type
    )
    syslog2(
        LOG_WARNING, 
        "bot core initialized", 
        profile=ctx.profile_manager.get_current_profile() if ctx.profile_manager else "unknown", 
        db_url=paths["db_url"], 
        vector=paths["vector_db_path"], 
        model=model_name
    )
    return bot_instance_local


def _register_profile_commands(admin_router_local: AdminCommandRouter, ctx: RuntimeContext) -> None:
    """Register profile commands."""
    profile_commands = ProfileCommands(ctx.profile_manager)
    _register_command_group(
        admin_router_local,
        "profile",
        profile_commands,
        {
            "list_profiles": "list",
            "create_profile": "create",
            "get_profile": "get",
            "set_profile": "set",
            "delete_profile": "delete",
            "profile_info": "info",
        }
    )


def _register_ingest_commands(admin_router_local: AdminCommandRouter, task_manager_local: TaskManager, ctx: RuntimeContext) -> IngestCommands:
    """Register ingest commands."""
    ingest_commands_local = IngestCommands(ctx.profile_manager, task_manager_local)
    _register_command_group(
        admin_router_local,
        "ingest",
        ingest_commands_local,
        {
            "start_ingest": None,
            "clear_data": "clear",
            "ingest_status": "status",
        }
    )
    return ingest_commands_local


def _register_stats_commands(admin_router_local: AdminCommandRouter, ctx: RuntimeContext) -> None:
    """Register stats commands."""
    stats_commands = StatsCommands(ctx.profile_manager)
    admin_router_local.register("stats", stats_commands.show_stats)
    admin_router_local.register("health", stats_commands.health_check)
    admin_router_local.register("logs", stats_commands.show_logs)


def _register_admin_commands(admin_router_local: AdminCommandRouter, bot_instance_local: LegaleBot, ctx: RuntimeContext) -> Tuple[TaskManager, IngestCommands]:
    """
    Register all admin commands in the router.
    
    Args:
        admin_router_local: AdminCommandRouter instance
        bot_instance_local: LegaleBot instance
        ctx: RuntimeContext instance
        
    Returns:
        Tuple of (task_manager, ingest_commands)
    """
    task_manager_local = TaskManager()

    # Register command groups
    _register_profile_commands(admin_router_local, ctx)
    ingest_commands_local = _register_ingest_commands(admin_router_local, task_manager_local, ctx)
    _register_stats_commands(admin_router_local, ctx)

    # control commands – сюда прокидываем колбэк hot-reload
    control_commands = ControlCommands(ctx.profile_manager, reload_callback=reload_for_current_profile)
    admin_router_local.register("restart", control_commands.restart_bot)

    # settings commands
    settings_commands = SettingsCommands(ctx.profile_manager)
    # Register manage_chats as direct handler so fallback passes subcommand in args
    admin_router_local.register("allowed", settings_commands.manage_chats)
    admin_router_local.register("allowed", settings_commands.lookup_chats, "lookup")
    admin_router_local.register("chat", settings_commands.manage_chats)
    admin_router_local.register("frequency", settings_commands.manage_frequency)

    # model commands
    model_commands = ModelCommands(ctx.profile_manager, bot_instance_local)
    _register_command_group(
        admin_router_local,
        "model",
        model_commands,
        {
            "list_models": "list",
            "get_model": "get",
            "set_model": "set",
        }
    )

    # system prompt commands
    system_prompt_commands = SystemPromptCommands(ctx.profile_manager)
    _register_command_group(
        admin_router_local,
        "system_prompt",
        system_prompt_commands,
        {
            "get_prompt": "get",
            "set_prompt": "set",
            "reset_prompt": "reset",
        }
    )

    # help command
    help_commands = HelpCommands()
    admin_router_local.register("help", help_commands.show_help)

    syslog2(LOG_INFO, "admin router initialized")
    
    return task_manager_local, ingest_commands_local


def _create_bot_instance(paths: Dict, admin_manager_local: AdminManager, ctx: RuntimeContext, args: Optional[SimpleNamespace] = None) -> Tuple[LegaleBot, bool]:
    """
    Create LegaleBot instance.
    
    Args:
        paths: Profile paths dictionary
        admin_manager_local: AdminManager instance
        ctx: RuntimeContext instance
        args: Optional namespace with configuration overrides
        
    Returns:
        Tuple of (Initialized LegaleBot instance, debug_rag flag)
    """
    # Get bot configuration
    model_name, debug_rag, log_level, retrieval_type = _get_bot_configuration(admin_manager_local, args)
    
    # Create LegaleBot
    profile_dir = paths["profile_dir"]
    bot_instance_local = _create_legale_bot(paths, model_name, log_level, debug_rag, profile_dir, retrieval_type, ctx=ctx)
    
    return bot_instance_local, debug_rag


def _create_admin_components(paths: Dict, bot_instance_local: LegaleBot, ctx: RuntimeContext) -> Tuple[AdminCommandRouter, TaskManager, IngestCommands]:
    """
    Create admin router, task manager and ingest commands.
    
    Args:
        paths: Profile paths dictionary
        bot_instance_local: LegaleBot instance
        ctx: RuntimeContext instance
        
    Returns:
        Tuple of (admin_router, task_manager, ingest_commands)
    """
    # Create admin router and register commands
    admin_router_local = AdminCommandRouter()
    task_manager_local, ingest_commands_local = _register_admin_commands(admin_router_local, bot_instance_local, ctx)
    
    return admin_router_local, task_manager_local, ingest_commands_local


def _create_command_dispatcher(
    bot_instance_local: LegaleBot,
    admin_manager_local: AdminManager,
    admin_router_local: AdminCommandRouter,
    debug_rag: bool
):
    """
    Create command dispatcher via CommandService.
    
    Args:
        bot_instance_local: LegaleBot instance
        admin_manager_local: AdminManager instance
        admin_router_local: AdminCommandRouter instance
        debug_rag: Debug RAG flag
        
    Returns:
        CommandDispatcher instance (from CommandService for backward compatibility)
    """
    from src.core.command_service import CommandService
    from src.app.main_cli import register_sync_handlers, register_async_handlers
    
    command_service = CommandService()
    register_sync_handlers(command_service, bot_instance_local, admin_manager_local, debug_rag)
    register_async_handlers(command_service, admin_manager_local, admin_router_local)
    
    syslog2(LOG_NOTICE, "command dispatcher initialized with admin handlers")
    return command_service.dispatcher


# инициализация рантайма под текущий профиль
async def init_runtime_for_current_profile(
    ctx: Optional[RuntimeContext] = None,
    args: Optional[SimpleNamespace] = None,
):
    """
    создать/переинициализировать bot_instance, admin_manager, admin_router и связанные команды
    под текущий активный профиль profile_manager
    """
    if ctx is None:
        ctx = get_runtime_context()

    # Step 1: Get profile paths
    paths = _get_profile_paths(ctx)

    # Step 2: Create admin manager (needed for bot configuration)
    profile_dir = paths["profile_dir"]
    admin_manager_local = _create_admin_manager(profile_dir)

    # Step 3: Create bot instance (also returns debug_rag flag)
    bot_instance_local, debug_rag = _create_bot_instance(paths, admin_manager_local, ctx, args)

    # Step 4: Create admin components (router, task_manager, ingest_commands)
    admin_router_local, task_manager_local, ingest_commands_local = _create_admin_components(paths, bot_instance_local, ctx)

    # Step 5: Create command dispatcher
    command_dispatcher_local = _create_command_dispatcher(
        bot_instance_local,
        admin_manager_local,
        admin_router_local,
        debug_rag
    )

    # только после успешного создания всех локальных объектов – публикуем их в runtime context
    ctx.bot_instance = bot_instance_local
    ctx.admin_manager = admin_manager_local
    ctx.admin_router = admin_router_local
    ctx.task_manager = task_manager_local
    ctx.ingest_commands = ingest_commands_local
    ctx.command_dispatcher = command_dispatcher_local

    return paths


# hot-reload рантайма под активный профиль (используется /admin restart)
async def reload_for_current_profile(args: Optional[SimpleNamespace] = None):
    """
    hot-reload рантайма под активный профиль (используется /admin restart)
    """
    syslog2(LOG_WARNING, "hot reload requested")
    paths = await init_runtime_for_current_profile(args=args)
    ctx = get_runtime_context()
    syslog2(LOG_WARNING, "hot reload completed", profile=ctx.profile_manager.get_current_profile() if ctx.profile_manager else "unknown", db=paths["db_path"], vector=paths["vector_db_path"])
    return paths


def _map_syslog2_to_logging_level(syslog2_level: int) -> int:
    """
    Map syslog2 constant to logging level.
    
    Args:
        syslog2_level: syslog2 log level constant
        
    Returns:
        logging level constant
    """
    mapping = {
        LOG_ALERT: logging.CRITICAL,
        LOG_CRIT: logging.CRITICAL,
        LOG_ERR: logging.ERROR,
        LOG_WARNING: logging.WARNING,
        LOG_NOTICE: logging.INFO,
        LOG_INFO: logging.INFO,
        LOG_DEBUG: logging.DEBUG,
    }
    return mapping.get(syslog2_level, logging.WARNING)


def _create_log_handler(use_syslog: bool) -> logging.Handler:
    """
    Create logging handler based on configuration.
    
    Args:
        use_syslog: If True, create SysLogHandler, otherwise StreamHandler
        
    Returns:
        Configured logging handler
    """
    if use_syslog:
        from logging.handlers import SysLogHandler
        return SysLogHandler(address='/dev/log')
    else:
        return logging.StreamHandler(sys.stdout)


def _setup_formatter(handler: logging.Handler, use_syslog: bool) -> None:
    """
    Setup formatter for logging handler.
    
    Args:
        handler: Logging handler to configure
        use_syslog: If True, use syslog format, otherwise use standard format
    """
    if use_syslog:
        formatter = logging.Formatter('legale-bot[%(process)d]: %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)


def _configure_loggers(handler: logging.Handler, level: int) -> None:
    """
    Configure root and uvicorn loggers with handler and level.
    
    Args:
        handler: Logging handler to add to loggers
        level: Logging level to set
    """
    # Configure root logger to capture all logs including syslog2 ("app")
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Also configure uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(level)


def setup_logging(log_level: Optional[str] = None, use_syslog: bool = False):
    """
    Configure logging based on log level.
    
    Args:
        log_level: Log level string (INFO, DEBUG, WARNING, etc.) or number (6=INFO, 7=DEBUG)
        use_syslog: If True, log to syslog instead of stdout
    """
    # Map log_level to logging level using shared mapping function
    if log_level:
        syslog2_level = _map_log_level_to_constant(log_level)
        level = _map_syslog2_to_logging_level(syslog2_level)
    else:
        level = logging.WARNING
    
    handler = _create_log_handler(use_syslog)
    _setup_formatter(handler, use_syslog)
    _configure_loggers(handler, level)


async def _init_profile_manager() -> None:
    """Initialize profile manager."""
    ctx = get_runtime_context()
    try:
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        
        # Import ProfileManager from legale.py
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from legale import ProfileManager
        
        # Create ProfileManager instance
        ctx.profile_manager = ProfileManager(project_root)
        
        logger.info("Profile manager initialized")
        syslog2(LOG_NOTICE, "profile manager initialized", profile=ctx.profile_manager.get_current_profile())
    except Exception as e:
        syslog2(LOG_ERR, "profile manager init failed", error=str(e))
        ctx.profile_manager = None
        raise RuntimeError("Profile manager initialization failed") from e


async def _init_runtime(args: Optional[SimpleNamespace] = None) -> None:
    """Initialize runtime for current profile."""
    try:
        await init_runtime_for_current_profile(args=args)
    except Exception as e:
        syslog2(LOG_ERR, "runtime init failed", error=str(e))
        raise


async def _init_telegram_app() -> None:
    """Initialize Telegram application."""
    ctx = get_runtime_context()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        syslog2(LOG_ERR, "telegram token missing")
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    ctx.telegram_app = Application.builder().token(token).build()
    await ctx.telegram_app.initialize()
    syslog2(LOG_NOTICE, "telegram app initialized")


@asynccontextmanager
async def lifespan(app: FastAPI, args: Optional[SimpleNamespace] = None):
    """
    Lifespan context manager for FastAPI.
    Loads bot instance on startup, cleans up on shutdown.
    """
    ctx = get_runtime_context()
    
    syslog2(LOG_NOTICE, "daemon starting")
    
    # Initialize components
    await _init_profile_manager()
    await _init_runtime(args)
    await _init_telegram_app()
    
    # Initialize access control service
    if ctx.admin_manager:
        ctx.access_control = AccessControlService(ctx.admin_manager)
        syslog2(LOG_NOTICE, "access control initialized")
    else:
        syslog2(LOG_WARNING, "access control not initialized", reason="admin_manager is None")
    
    yield
    
    # Cleanup
    syslog2(LOG_NOTICE, "shutting down")
    if ctx.telegram_app:
        await ctx.telegram_app.shutdown()
    syslog2(LOG_NOTICE, "shutdown complete")


async def process_document_update(update: Update) -> Optional[str]:
    """
    Process document update (file upload for ingestion).
    
    Args:
        update: Telegram update object
        
    Returns:
        Response text to send to user, or None if no response needed
    """
    ctx = get_runtime_context()
    if not (update.message and update.message.document and ctx.ingest_commands):
        return None
    
    return await ctx.ingest_commands.handle_file_upload(update, None, ctx.admin_manager)


async def process_text_update(update: Update) -> None:
    """
    Process text message update.
    
    Args:
        update: Telegram update object
    """
    if not (update.message and update.message.text):
        return
    
    try:
        await handle_message(update)
    except Exception as e:
        syslog2(LOG_ERR, "handle_message failed", error=str(e), update_id=update.update_id)
        # Try to send error message to user
        try:
            ctx = get_runtime_context()
            if update.message:
                await ctx.telegram_app.bot.send_message(
                    chat_id=update.message.chat_id,
                    text="Произошла ошибка при обработке сообщения. Попробуйте позже."
                )
        except:
            pass


async def _validate_webhook_request(request: Request) -> Optional[dict]:
    """
    Validate webhook request body and extract JSON data.
    
    Args:
        request: FastAPI request object
        
    Returns:
        JSON data as dict or None if validation failed
    """
    try:
        data = await request.json()
        if not isinstance(data, dict):
            syslog2(LOG_ERR, "webhook request body is not a dict", body_type=type(data).__name__)
            return None
        return data
    except Exception as e:
        syslog2(LOG_ERR, "webhook request validation failed", error=str(e))
        return None


def _parse_update_from_json(data: dict, bot) -> Optional[Update]:
    """
    Parse Update object from JSON data.
    
    Args:
        data: JSON data as dict
        bot: Telegram bot instance
        
    Returns:
        Update object or None if parsing failed
    """
    try:
        update = Update.de_json(data, bot)
        if update is None:
            syslog2(LOG_ERR, "failed to parse Update from JSON data")
            return None
        syslog2(LOG_DEBUG, "update received", update_id=update.update_id)
        return update
    except Exception as e:
        syslog2(LOG_ERR, "webhook parse failed", error=str(e))
        return None


async def _parse_webhook_update(request: Request, ctx: Optional[RuntimeContext] = None) -> Optional[Update]:
    """
    Parse Telegram update from HTTP request.
    
    Args:
        request: FastAPI request object
        ctx: Optional RuntimeContext (for tests/injection)
        
    Returns:
        Update object or None if parsing failed
    """
    # Step 1: Validate request body
    data = await _validate_webhook_request(request)
    if data is None:
        return None
    
    # Step 2: Parse Update from JSON
    if ctx is None:
        ctx = get_runtime_context()
    update = _parse_update_from_json(data, ctx.telegram_app.bot)
    return update


async def _handle_command(update: Update) -> None:
    """
    Handle command message.
    
    Args:
        update: Telegram update object
    """
    message = update.message
    text = message.text
    chat_id = message.chat_id
    user_id = message.from_user.id
    is_private = (message.chat.type == "private")
    
    syslog2(LOG_NOTICE, "command received", chat_id=chat_id, user_id=user_id, command=text[:50])
    
    # Step 1: Handle public commands (bypass access control)
    if await _handle_public_commands_step(message, text, chat_id):
        return
    
    # Step 2: Check access
    if not await _check_access_step(user_id, chat_id, is_private, True, text):
        return
    
    # Step 3: Determine if bot should respond
    respond, reason = await _determine_response_step(message, True, is_private, chat_id)
    
    # Step 4: Check if bot_instance is available
    ctx = get_runtime_context()
    if not ctx.bot_instance:
        syslog2(LOG_ERR, "bot instance missing", action="drop_message")
        return
    
    # Step 5: Route command to appropriate handler
    handler = MessageHandler(ctx.bot_instance, ctx.admin_manager, ctx.admin_router, ctx)
    response = await _process_command_message(text, update, handler, respond)
    if response is None:
        return  # Command was not recognized or ignored
    
    # Step 6: Send response if available
    await _send_response_if_available(response, chat_id, True, respond)


async def _handle_user_message(update: Update) -> None:
    """
    Handle regular user message (non-command).
    
    Args:
        update: Telegram update object
    """
    message = update.message
    text = message.text
    chat_id = message.chat_id
    user_id = message.from_user.id
    is_private = (message.chat.type == "private")
    
    syslog2(LOG_NOTICE, "user message received", chat_id=chat_id, user_id=user_id, text_snippet=text[:50])
    
    # Step 1: Check access
    if not await _check_access_step(user_id, chat_id, is_private, False, None):
        return
    
    # Step 2: Determine if bot should respond
    respond, reason = await _determine_response_step(message, False, is_private, chat_id)
    
    # Step 3: Check if bot_instance is available
    ctx = get_runtime_context()
    if not ctx.bot_instance:
        syslog2(LOG_ERR, "bot instance missing", action="drop_message")
        return
    
    # Step 4: Parse and handle search mentions
    bot_username = (ctx.telegram_app.bot.username or "").lower()
    bot_id = ctx.telegram_app.bot.id
    if await _handle_search_mention_step(message, bot_username, bot_id, chat_id):
        return
    
    # Step 5: Remove mention token from text if present
    if text:
        extracted_text = _extract_mention_text(text, bot_username)
        if extracted_text is not None:
            text = extracted_text
    
    # Step 6: Route message to appropriate handler
    handler = MessageHandler(ctx.bot_instance, ctx.admin_manager, ctx.admin_router, ctx)
    response = await _process_regular_message(text, handler, respond, ctx.admin_manager.config, chat_id)
    if response is None:
        return  # Message was ignored
    
    # Step 7: Send response if available
    await _send_response_if_available(response, chat_id, False, respond)


async def _process_webhook_update(update: Update, ctx: Optional[RuntimeContext] = None) -> Optional[str]:
    """
    Process Telegram update.
    
    Args:
        update: Telegram update object
        
    Returns:
        Response text to send (backward-compatible, primarily for document updates) or None
    """
    if ctx is not None:
        if getattr(update, "message", None) and getattr(update.message, "document", None):
            return await process_document_update(update)
        await process_text_update(update)
        return None

    parsed_update = parse_update(update)
    request = _to_app_request(parsed_update)
    app = _get_telegram_transport_app()
    response = await app.handle_request(request)
    await _send_response(parsed_update, response)
    return response.text or None


def parse_update(update: Update) -> Update:
    """Parse/validate incoming Update for the transport."""
    return update


def _to_app_request(update: Update) -> AppRequest:
    """Map Telegram Update to AppRequest."""
    message = getattr(update, "message", None)
    user_id = None
    chat_id = None
    text = ""
    if message:
        try:
            user = getattr(message, "from_user", None)
            if user and getattr(user, "id", None) is not None:
                user_id = str(user.id)
        except Exception:
            user_id = None
        try:
            if getattr(message, "chat_id", None) is not None:
                chat_id = str(message.chat_id)
        except Exception:
            chat_id = None
        text = getattr(message, "text", "") or ""

    return AppRequest(
        user_id=user_id,
        chat_id=chat_id,
        text=text,
        transport="telegram",
        meta={"update": update},
    )


async def _send_response(update: Update, response: AppResponse) -> None:
    """Send AppResponse to Telegram."""
    if not response.text:
        return
    message = getattr(update, "message", None)
    if not message:
        return
    ctx = get_runtime_context()
    telegram_app = getattr(ctx, "telegram_app", None)
    bot = getattr(telegram_app, "bot", None)
    if not bot:
        return
    result = bot.send_message(chat_id=message.chat_id, text=response.text)
    if inspect.isawaitable(result):
        await result


class _TelegramTransportApp:
    async def handle_request(self, request: AppRequest) -> AppResponse:
        update = (request.meta or {}).get("update")
        if update is None:
            return AppResponse(text="")

        response_text = await process_document_update(update)
        if response_text:
            return AppResponse(text=response_text)

        await process_text_update(update)
        return AppResponse(text="")


_telegram_transport_app: Optional[_TelegramTransportApp] = None


def _get_telegram_transport_app() -> _TelegramTransportApp:
    global _telegram_transport_app
    if _telegram_transport_app is None:
        _telegram_transport_app = _TelegramTransportApp()
    return _telegram_transport_app


def _create_webhook_handler():
    """
    Create webhook endpoint handler function.
    
    Returns:
        Async function that handles webhook requests
    """
    async def webhook(request: Request):
        """
        Handle incoming Telegram webhook updates.
        HTTP endpoint that parses requests and delegates to business logic.
        """
        # Parse update from HTTP request
        update = await _parse_webhook_update(request)
        if update is None:
            return Response(status_code=400)
        
        try:
            # Process update (business logic)
            await _process_webhook_update(update)
            
            return Response(status_code=200)
        
        except Exception as e:
            syslog2(LOG_ERR, "webhook processing failed", error=str(e))
            return Response(status_code=500)
    
    return webhook

def _setup_webhook_endpoint(app: FastAPI):
    """
    Setup webhook endpoint for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    webhook_handler = _create_webhook_handler()
    app.post("/webhook")(webhook_handler)


def create_app(args: Optional[SimpleNamespace] = None):
    """
    Create FastAPI app with lifespan that captures args.
    """
    # Create lifespan with closure on args
    @asynccontextmanager
    async def lifespan_with_args(app: FastAPI):
        async with lifespan(app, args):
            yield
    
    app = FastAPI(lifespan=lifespan_with_args)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        ctx = get_runtime_context()
        return {"status": "healthy", "bot_loaded": ctx.bot_instance is not None}
    
    # Setup webhook endpoint
    _setup_webhook_endpoint(app)
    
    return app


def _check_mention_entity(ent, text: str, bot_username: str) -> bool:
    """Check if mention entity matches bot username."""
    try:
        mention_text = text[ent.offset: ent.offset + ent.length]
        return bot_username and mention_text.lower() == f"@{bot_username}"
    except Exception:
        return False


def _check_text_mention_entity(ent, bot_id: int) -> bool:
    """Check if text_mention entity matches bot id."""
    try:
        return ent.user and ent.user.id == bot_id
    except Exception:
        return False


def is_bot_mentioned(message, bot_username: str, bot_id: int) -> bool:
    """
    check if bot is mentioned in message (by @username or text_mention)
    """
    text = message.text or ""
    entities = getattr(message, "entities", None) or []
    if not entities:
        return False

    if not isinstance(entities, (list, tuple)):
        try:
            entities = list(entities)
        except TypeError:
            return False

    for ent in entities:
        try:
            if ent.type == "mention":
                if _check_mention_entity(ent, text, bot_username):
                    return True
            elif ent.type == "text_mention":
                if _check_text_mention_entity(ent, bot_id):
                    return True
        except Exception:
            continue

    return False


def _check_admin_manager(ctx: RuntimeContext) -> bool:
    """
    Check if admin_manager is available.
    
    Args:
        ctx: RuntimeContext instance
        
    Returns:
        True if admin_manager is available, False otherwise
    """
    if not ctx.admin_manager:
        syslog2(LOG_ERR, "admin manager missing", action="drop_message")
        return False
    return True


def _check_bot_instance(ctx: RuntimeContext) -> bool:
    """
    Check if bot_instance is available.
    
    Args:
        ctx: RuntimeContext instance
        
    Returns:
        True if bot_instance is available, False otherwise
    """
    if not ctx.bot_instance:
        syslog2(LOG_ERR, "bot instance missing", action="drop_message")
        return False
    return True


def _ensure_required_components() -> Optional[MessageHandler]:
    """
    Ensure admin_manager and bot_instance are available and create MessageHandler.
    
    Returns:
        MessageHandler instance if components are available, None otherwise
    """
    ctx = get_runtime_context()
    if not _check_admin_manager(ctx):
        return None
    
    if not _check_bot_instance(ctx):
        return None
    
    return MessageHandler(ctx.bot_instance, ctx.admin_manager, ctx.admin_router)


async def _execute_handler(handler_func, handler: MessageHandler, *args, **kwargs) -> Optional[str]:
    """
    Execute handler function and return response.
    
    Args:
        handler_func: Async function that takes MessageHandler and returns response string
        handler: MessageHandler instance
        *args, **kwargs: Additional arguments to pass to handler_func
        
    Returns:
        Response string from handler, or None if no response
    """
    return await handler_func(handler, *args, **kwargs)


async def _send_handler_response(response: str, chat_id: int) -> None:
    """
    Send handler response to chat.
    
    Args:
        response: Response text to send
        chat_id: Chat ID for sending response
    """
    ctx = get_runtime_context()
    await ctx.telegram_app.bot.send_message(chat_id=chat_id, text=response)


async def _ensure_handler_available(handler_func, chat_id: int, *args, **kwargs) -> bool:
    """
    Ensure MessageHandler is available, execute handler function and send response.
    
    Args:
        handler_func: Async function that takes MessageHandler and returns response string
        chat_id: Chat ID for sending response
        *args, **kwargs: Additional arguments to pass to handler_func
        
    Returns:
        True if handler was executed and response sent, False if handler unavailable
    """
    handler = _ensure_required_components()
    if not handler:
        return True  # Signal that command was "handled" (by dropping it)
    
    response = await _execute_handler(handler_func, handler, *args, **kwargs)
    if response:
        await _send_handler_response(response, chat_id)
    return True


async def _handle_id_command(message, chat_id: int) -> bool:
    """
    Handle /id command - send chat ID and user ID.
    
    Args:
        message: Telegram message object
        chat_id: Chat ID
        
    Returns:
        True (command was handled)
    """
    user_id = message.from_user.id
    ctx = get_runtime_context()
    await ctx.telegram_app.bot.send_message(
        chat_id=chat_id,
        text=f"Chat ID: `{chat_id}`\nUser ID: `{user_id}`",
        parse_mode="Markdown",
    )
    return True


async def _handle_help_command(text: str, chat_id: int) -> bool:
    """
    Handle /help command.
    
    Args:
        text: Message text
        chat_id: Chat ID
        
    Returns:
        True if command was handled, False otherwise
    """
    if text == "/help" or (text.startswith("/") and text.startswith("/help")):
        async def help_handler(handler: MessageHandler) -> str:
            return await handler.handle_help_command()
        return await _ensure_handler_available(help_handler, chat_id)
    return False


async def _handle_admin_set_command(text: str, message, chat_id: int) -> bool:
    """
    Handle /admin_set or /set_admin command.
    
    Args:
        text: Message text
        message: Telegram message object
        chat_id: Chat ID
        
    Returns:
        True if command was handled, False otherwise
    """
    if text.startswith("/") and (text.startswith("/admin_set") or text.startswith("/set_admin")):
        # Normalize command to /admin_set for handler
        normalized_text = text.replace("/set_admin", "/admin_set", 1) if text.startswith("/set_admin") else text
        
        async def admin_set_handler(handler: MessageHandler) -> str:
            return await handler.handle_admin_set_command(normalized_text, message)
        return await _ensure_handler_available(admin_set_handler, chat_id)
    return False


async def _handle_public_commands(message, text: str, chat_id: int) -> bool:
    """
    Handle public commands that bypass access control.
    
    Args:
        message: Telegram message object
        text: Message text
        chat_id: Chat ID
        
    Returns:
        True if command was handled, False otherwise
    """
    # Handle /id command
    if text == "/id":
        return await _handle_id_command(message, chat_id)
    
    # Handle /help command
    if await _handle_help_command(text, chat_id):
        return True
    
    # Handle /admin_set or /set_admin command
    if await _handle_admin_set_command(text, message, chat_id):
        return True
    
    return False


def _check_access(user_id: int, chat_id: int, is_private: bool, is_command: bool, command_text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if user has access to send message.
    
    Args:
        user_id: User ID
        chat_id: Chat ID
        is_private: Whether message is from private chat
        is_command: Whether message is a command
        command_text: Command text if is_command is True
        
    Returns:
        Tuple of (is_allowed, denial_reason)
    """
    ctx = get_runtime_context()
    if not ctx.admin_manager:
        syslog2(LOG_ERR, "admin manager missing", action="drop_message")
        return False, "admin_manager missing"
    
    if not ctx.access_control:
        syslog2(LOG_ERR, "access control missing")
        return False, "access_control missing"
    
    is_allowed, denial_reason = ctx.access_control.is_allowed(
        user_id=user_id,
        chat_id=chat_id,
        is_private=is_private,
        is_command=is_command,
        command_text=command_text
    )
    
    if not is_allowed:
        syslog2(LOG_DEBUG, "access denied", chat_id=chat_id, user_id=user_id, reason=denial_reason)
    
    return is_allowed, denial_reason


def _extract_mention_text(text: str, bot_username: str) -> Optional[str]:
    """
    Extract text after bot mention prefix.
    
    Args:
        text: Message text
        bot_username: Bot username (lowercase)
        
    Returns:
        Text after mention prefix, or None if mention not found
    """
    raw_text = text.strip()
    lowered = raw_text.lower()
    mention_prefix = f"@{bot_username}"
    
    if not lowered.startswith(mention_prefix):
        return None
    
    return raw_text[len(mention_prefix):].lstrip()


def _parse_search_command(text_after_mention: str) -> Optional[str]:
    """
    Parse search command from text after mention.
    
    Args:
        text_after_mention: Text after bot mention
        
    Returns:
        Search query if valid search command found, None otherwise
    """
    parts = text_after_mention.split(maxsplit=1)
    
    if not parts:
        return None
    
    first = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    
    if first in ("поиск", "find") and rest:
        return rest
    
    return None


def _parse_search_mention(message, bot_username: str, bot_id: int) -> Tuple[bool, str]:
    """
    Parse search mention from message.
    
    Args:
        message: Telegram message object
        bot_username: Bot username (lowercase)
        bot_id: Bot ID
        
    Returns:
        Tuple of (is_search_command, search_query)
    """
    text = message.text or ""
    has_mention = is_bot_mentioned(message, bot_username, bot_id)
    is_command = text.startswith("/")
    
    if not has_mention or is_command:
        return False, ""
    
    text_after_mention = _extract_mention_text(text, bot_username)
    if text_after_mention is None:
        return False, ""
    
    search_query = _parse_search_command(text_after_mention)
    if search_query is None:
        return False, ""
    
    return True, search_query


async def _send_single_message_part(chat_id: int, part: Dict) -> None:
    """
    Send a single message part to chat.
    
    Args:
        chat_id: Chat ID
        part: Message part dictionary with "content" key
    """
    ctx = get_runtime_context()
    await ctx.telegram_app.bot.send_message(
        chat_id=chat_id,
        text=part["content"],
        parse_mode="HTML"
    )


async def _handle_send_error(chat_id: int, error: Exception, context: str = "message parts") -> None:
    """
    Handle error when sending message.
    
    Args:
        chat_id: Chat ID
        error: Exception that occurred
        context: Context description for logging (e.g., "empty message", "message parts")
    """
    syslog2(LOG_ERR, f"failed to send {context}", chat_id=chat_id, error=str(error))


async def _send_message_parts_unified(chat_id: int, message_parts_list: List[List[Dict]], empty_message: str = "", log_context: Dict = None) -> int:
    """
    Unified helper for sending message parts with logging.
    
    Args:
        chat_id: Chat ID
        message_parts_list: List of message parts to send
        empty_message: Message to send if message_parts_list is empty
        log_context: Additional context for logging
        
    Returns:
        Total number of message parts sent
    """
    ctx = get_runtime_context()
    if not message_parts_list:
        if empty_message:
            try:
                await ctx.telegram_app.bot.send_message(chat_id=chat_id, text=empty_message)
            except Exception as e:
                await _handle_send_error(chat_id, e, "empty message")
        return 0
    
    # Send each message part as separate message
    total_parts = 0
    try:
        for message_parts in message_parts_list:
            for part in message_parts:
                await _send_single_message_part(chat_id, part)
                total_parts += 1
        
        log_data = {"chat_id": chat_id, "messages": len(message_parts_list), "parts": total_parts}
        if log_context:
            log_data.update(log_context)
        syslog2(LOG_NOTICE, "message parts sent", **log_data)
    except Exception as e:
        await _handle_send_error(chat_id, e, "message parts")
    
    return total_parts


async def _send_message_parts(chat_id: int, message_parts_list: List[List[Dict]], empty_message: str = "", log_context: Dict = None) -> int:
    """
    Send message parts to chat and return total number of parts sent.
    
    Deprecated: Use _send_message_parts_unified instead.
    
    Args:
        chat_id: Chat ID
        message_parts_list: List of message parts to send
        empty_message: Message to send if message_parts_list is empty
        log_context: Additional context for logging
        
    Returns:
        Total number of message parts sent
    """
    return await _send_message_parts_unified(chat_id, message_parts_list, empty_message, log_context)


async def _send_search_results(chat_id: int, message_parts_list: List[List[Dict]], query: str) -> None:
    """
    Send search results to chat.
    
    Args:
        chat_id: Chat ID
        message_parts_list: List of message parts from search
        query: Search query
    """
    empty_message = f'по запросу "{query}" ничего не найдено' if query else ""
    await _send_message_parts_unified(
        chat_id,
        message_parts_list,
        empty_message,
        {"query": query}
    )


async def _should_ignore_message(respond: bool, target_freq: int, chat_id: int) -> bool:
    """
    Check if message should be ignored based on frequency settings.
    
    Args:
        respond: Whether bot should respond
        target_freq: Response frequency setting
        chat_id: Chat ID for logging
        
    Returns:
        True if message should be ignored
    """
    if not respond and target_freq == 0:
        syslog2(LOG_DEBUG, "message ignored", chat_id=chat_id, reason="freq=0, no mention")
        return True
    return False


async def _handle_search_mention(message, bot_username: str, bot_id: int, chat_id: int) -> bool:
    """
    Handle search mention command.
    
    Args:
        message: Telegram message object
        bot_username: Bot username (lowercase)
        bot_id: Bot ID
        chat_id: Chat ID
        
    Returns:
        True if search mention was handled, False otherwise
    """
    is_search_command, search_query = _parse_search_mention(message, bot_username, bot_id)
    if is_search_command:
        ctx = get_runtime_context()
        message_parts_list = search_message_contents(ctx.bot_instance.retrieval, ctx.bot_instance.db, search_query, top_k=3)
        await _send_search_results(chat_id, message_parts_list, search_query)
        return True
    return False


async def _process_command_message(text: str, update: Update, handler: MessageHandler, respond: bool) -> Optional[str]:
    """
    Process command message.
    
    Args:
        text: Message text
        update: Telegram update object
        handler: MessageHandler instance
        respond: Whether bot should respond
        
    Returns:
        Response text or None
    """
    response = await handler.route_command(text, update)
    if response is None:
        # Not a recognized command, treat as regular query
        syslog2(LOG_ALERT, "handle_user_query as not a recognized command", text=text, respond=respond)
        response = await handler.handle_user_query(text, respond)
    return response


async def _process_regular_message(text: str, handler: MessageHandler, respond: bool, config, chat_id: int) -> Optional[str]:
    """
    Process regular (non-command) message.
    
    Args:
        text: Message text
        handler: MessageHandler instance
        respond: Whether bot should respond
        config: Admin config
        chat_id: Chat ID
        
    Returns:
        Response text or None if message should be ignored
    """
    target_freq = config.response_frequency or 0
    if await _should_ignore_message(respond, target_freq, chat_id):
        return None

    syslog2(LOG_ALERT, "handle_user_query as regular query", text=text, respond=respond)
    return await handler.handle_user_query(text, respond)


async def _determine_response_decision(message, is_command: bool, is_private: bool, chat_id: int) -> Tuple[bool, str]:
    """
    Determine if bot should respond to message.
    
    Args:
        message: Telegram message object
        is_command: Whether message is a command
        is_private: Whether message is from private chat
        chat_id: Chat ID
        
    Returns:
        Tuple of (should_respond, reason)
    """
    ctx = get_runtime_context()
    config = ctx.admin_manager.config
    bot_username = (ctx.telegram_app.bot.username or "").lower()
    bot_id = ctx.telegram_app.bot.id
    has_mention = is_bot_mentioned(message, bot_username, bot_id)
    
    respond, reason = ctx.frequency_controller.should_respond(
        chat_id=chat_id,
        frequency=config.response_frequency or 0,
        has_mention=has_mention,
        is_command=is_command,
        is_private=is_private
    )
    
    syslog2(LOG_DEBUG, "response decision", chat_id=chat_id, respond=respond, reason=reason, is_command=is_command, is_private=is_private, has_mention=has_mention)
    return respond, reason


async def _send_response_if_available(response: Optional[str], chat_id: int, is_command: bool, respond: bool) -> None:
    """
    Send response message if available.
    
    Args:
        response: Response text or None
        chat_id: Chat ID
        is_command: Whether original message was a command
        respond: Whether bot should respond
    """
    if response:
        try:
            ctx = get_runtime_context()
            await ctx.telegram_app.bot.send_message(chat_id=chat_id, text=response)
            syslog2(LOG_NOTICE, "response sent", chat_id=chat_id, response_length=len(response))
        except Exception as e:
            syslog2(LOG_ERR, "failed to send response", chat_id=chat_id, error=str(e))
    else:
        syslog2(LOG_DEBUG, "no response generated", chat_id=chat_id, is_command=is_command, respond=respond)


async def _handle_public_commands_step(message, text: str, chat_id: int) -> bool:
    """
    Step 1: Handle public commands (bypass access control).
    
    Returns:
        True if command was handled and processing should stop, False otherwise
    """
    return await _handle_public_commands(message, text, chat_id)


async def _check_access_step(user_id: int, chat_id: int, is_private: bool, is_command: bool, command_text: Optional[str]) -> bool:
    """
    Step 2: Check if user has access to send message.
    
    Returns:
        True if access granted, False otherwise
    """
    is_allowed, _ = _check_access(user_id, chat_id, is_private, is_command, command_text)
    return is_allowed


async def _determine_response_step(message, is_command: bool, is_private: bool, chat_id: int) -> Tuple[bool, str]:
    """
    Step 3: Determine if bot should respond to message.
    
    Returns:
        Tuple of (should_respond, reason)
    """
    return await _determine_response_decision(message, is_command, is_private, chat_id)


async def _handle_search_mention_step(message, bot_username: str, bot_id: int, chat_id: int) -> bool:
    """
    Step 5: Parse and handle search mentions.
    
    Returns:
        True if search mention was handled, False otherwise
    """
    return await _handle_search_mention(message, bot_username, bot_id, chat_id)


async def _route_message_step(text: str, update: Update, is_command: bool, respond: bool, chat_id: int) -> Optional[str]:
    """
    Step 6: Route message to appropriate handler.
    
    Returns:
        Response text or None if message should be ignored
    """
    ctx = get_runtime_context()
    handler = MessageHandler(ctx.bot_instance, ctx.admin_manager, ctx.admin_router, ctx)
    
    if is_command:
        return await _process_command_message(text, update, handler, respond)
    else:
        return await _process_regular_message(text, handler, respond, ctx.admin_manager.config, chat_id)


async def handle_message(update: Update):
    """
    Process incoming text messages.
    
    Simplified version using MessageHandler and utility classes.
    """
    message = update.message
    text = message.text
    chat_id = message.chat_id
    user_id = message.from_user.id

    syslog2(LOG_NOTICE, "message received", chat_id=chat_id, user_id=user_id, text_snippet=text[:50])

    is_command = text.startswith("/")
    is_private = (message.chat.type == "private")

    # Step 1: Handle public commands (bypass access control)
    if await _handle_public_commands_step(message, text, chat_id):
        return

    # Step 2: Check access
    if not await _check_access_step(user_id, chat_id, is_private, is_command, text if is_command else None):
        return

    # Step 3: Determine if bot should respond
    respond, reason = await _determine_response_step(message, is_command, is_private, chat_id)

    # Step 4: Check if bot_instance is available
    ctx = get_runtime_context()
    if not ctx.bot_instance:
        syslog2(LOG_ERR, "bot instance missing", action="drop_message")
        return
    
    # Step 5: Parse and handle search mentions
    bot_username = (ctx.telegram_app.bot.username or "").lower()
    bot_id = ctx.telegram_app.bot.id
    if await _handle_search_mention_step(message, bot_username, bot_id, chat_id):
        return
    
    # Remove mention token from text if present (for non-commands)
    if not is_command and text:
        extracted_text = _extract_mention_text(text, bot_username)
        if extracted_text is not None:
            text = extracted_text
    
    # Step 6: Route message to appropriate handler
    response = await _route_message_step(text, update, is_command, respond, chat_id)
    if response is None:
        return  # Message was ignored

    # Step 7: Send response if available
    await _send_response_if_available(response, chat_id, is_command, respond)


def register_webhook(url: str, token: str):
    """
    Register webhook with Telegram.
    """
    import requests
    
    api_url = f"https://api.telegram.org/bot{token}/setWebhook"
    response = requests.post(api_url, json={"url": url})
    
    if response.status_code == 200:
        result = response.json()
        if result.get("ok"):
            print(f"Webhook registered successfully: {url}")
            print(f"  Description: {result.get('description', 'N/A')}")
        else:
            print(f"Failed to register webhook: {result.get('description')}")
            sys.exit(1)
    else:
        print(f"HTTP error: {response.status_code}")
        sys.exit(1)


def delete_webhook(token: str):
    """
    Delete webhook from Telegram.
    """
    import requests
    
    api_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    response = requests.post(api_url)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("ok"):
            print("Webhook deleted successfully")
        else:
            print(f"Failed to delete webhook: {result.get('description')}")
            sys.exit(1)
    else:
        print(f"HTTP error: {response.status_code}")
        sys.exit(1)


def _create_fastapi_app(args: Optional[SimpleNamespace] = None) -> FastAPI:
    """
    Create FastAPI application instance.
    
    Args:
        args: Parsed command line arguments (SimpleNamespace)
        
    Returns:
        FastAPI application instance
    """
    return create_app(args)

def _setup_server_logging(log_level: Optional[str] = None) -> Tuple[str, bool]:
    """
    Setup logging for server and map log level for uvicorn.
    
    Args:
        log_level: Log level string (INFO, DEBUG, WARNING, etc.) or number (6=INFO, 7=DEBUG)
        
    Returns:
        Tuple of (uvicorn_log_level, access_log)
    """
    setup_logging(log_level=log_level, use_syslog=False)
    
    # Map log_level to uvicorn log level (lowercase string)
    if log_level:
        # Convert to string if it's a number
        if not isinstance(log_level, str):
            log_level = str(log_level)
        log_level_upper = log_level.upper()
        uvicorn_level_map = {
            "1": "critical",  # ALERT
            "2": "critical",  # CRIT
            "3": "error",     # ERR
            "4": "warning",   # WARNING
            "5": "info",      # NOTICE
            "6": "info",      # INFO
            "7": "debug",     # DEBUG
            "ALERT": "critical",
            "CRIT": "critical",
            "ERR": "error",
            "ERROR": "error",
            "WARNING": "warning",
            "NOTICE": "info",
            "INFO": "info",
            "DEBUG": "debug",
        }
        uvicorn_log_level = uvicorn_level_map.get(log_level_upper, "warning")
        access_log = log_level_upper in ("6", "7", "INFO", "DEBUG")
    else:
        uvicorn_log_level = "warning"
        access_log = False
    
    return uvicorn_log_level, access_log

def _start_uvicorn_server(app: FastAPI, host: str, port: int, uvicorn_log_level: str, access_log: bool) -> None:
    """
    Start uvicorn server with given configuration.
    
    Args:
        app: FastAPI application instance
        host: Host to bind
        port: Port to bind
        uvicorn_log_level: Uvicorn log level (lowercase string)
        access_log: Whether to enable access log
    """
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        access_log=access_log
    )

def run_server(host: str = "127.0.0.1", port: int = 8000, log_level: Optional[str] = None, debug_rag: bool = False, args: Optional[SimpleNamespace] = None):
    """
    Run the FastAPI server in foreground mode.
    
    Args:
        host: Host to bind
        port: Port to bind
        log_level: Log level string (INFO, DEBUG, WARNING, etc.) or number (6=INFO, 7=DEBUG)
        debug_rag: Enable RAG debug mode
        args: Parsed command line arguments (SimpleNamespace)
    """
    _set_debug_rag_mode(debug_rag)
    
    uvicorn_log_level, access_log = _setup_server_logging(log_level)
    app = _create_fastapi_app(args)
    _start_uvicorn_server(app, host, port, uvicorn_log_level, access_log)


def _setup_signal_handlers() -> Dict[int, Callable]:
    """
    Setup signal handlers for daemon process.
    
    Returns:
        Dictionary mapping signal numbers to handler functions
    """
    return {
        signal.SIGTERM: lambda signum, frame: sys.exit(0),
        signal.SIGINT: lambda signum, frame: sys.exit(0),
    }


def _create_daemon_context(pid_file: str):
    """
    Create daemon context with signal handlers.
    
    Args:
        pid_file: Path to PID file
        
    Returns:
        DaemonContext instance
    """
    import daemon
    from daemon import pidfile
    
    signal_map = _setup_signal_handlers()
    return daemon.DaemonContext(
        pidfile=pidfile.TimeoutPIDLockFile(pid_file),
        signal_map=signal_map
    )


def run_daemon(host: str = "127.0.0.1", port: int = 8000, args: Optional[SimpleNamespace] = None):
    """
    Run the FastAPI server in daemon mode (background).
    
    Args:
        host: Host to bind
        port: Port to bind
        args: Parsed command line arguments (SimpleNamespace)
    """
    pid_file = "/var/run/legale-bot.pid"
    
    # Setup syslog logging
    setup_logging(log_level="INFO", use_syslog=True)
    
    # Create app with args
    app = create_app(args)
    
    # Create daemon context and run server
    with _create_daemon_context(pid_file):
        syslog2(LOG_NOTICE, "daemon started")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )


def _parse_cli_arguments():
    """
    Parse CLI arguments and return parsed arguments object.
    
    Returns:
        SimpleNamespace with parsed arguments including bot_command attribute
        
    Exits:
        If help requested or on parse errors
    """
    from src.lib.argparse2 import parse
    from types import SimpleNamespace
    
    # Build opt_table for global options (only -h, --help, -V)
    opt_table = {
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "V": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
    }
    
    try:
        # Parse global options first
        opts, remaining_args = parse(sys.argv[1:], opt_table)
        
        # Check for help
        if opts.get("h") or opts.get("help") or not remaining_args:
            print("Telegram Bot Webhook Server")
            print("\nCommands:")
            print("  register url <url> [token <token>] - Register webhook with Telegram")
            print("  delete [token <token>] - Delete webhook from Telegram")
            print("  run [host <host>] [port <port>] [token <token>] [debug-rag] - Run server in foreground")
            print("  daemon [host <host>] [port <port>] [token <token>] - Run server as daemon")
            sys.exit(0)
        
        # Extract command
        cmd = remaining_args[0]
        args_list = remaining_args[1:]
        
        # Parse command-specific arguments (words without dashes)
        parsed_opts = {}
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg == "url" and i + 1 < len(args_list):
                parsed_opts["url"] = args_list[i + 1]
                i += 2
            elif arg == "token" and i + 1 < len(args_list):
                parsed_opts["token"] = args_list[i + 1]
                i += 2
            elif arg == "host" and i + 1 < len(args_list):
                parsed_opts["host"] = args_list[i + 1]
                i += 2
            elif arg == "port" and i + 1 < len(args_list):
                parsed_opts["port"] = args_list[i + 1]
                i += 2
            elif arg == "debug-rag":
                parsed_opts["debug-rag"] = True
                i += 1
            else:
                i += 1
        
        # Parse command-specific arguments
        if cmd == "register":
            url = parsed_opts.get("url")
            if not url:
                print("Error: url required for bot register", file=sys.stderr)
                sys.exit(1)
            token = parsed_opts.get("token")
            parsed_args = SimpleNamespace(url=url, token=token, bot_command="register")
        
        elif cmd == "delete":
            token = parsed_opts.get("token")
            parsed_args = SimpleNamespace(token=token, bot_command="delete")
        
        elif cmd == "run":
            host = parsed_opts.get("host", "127.0.0.1")
            port = int(parsed_opts.get("port", 8000)) if parsed_opts.get("port") else 8000
            token = parsed_opts.get("token")
            debug_rag = parsed_opts.get("debug-rag", False)
            parsed_args = SimpleNamespace(host=host, port=port, token=token, debug_rag=debug_rag, bot_command="run")
        
        elif cmd == "daemon":
            host = parsed_opts.get("host", "127.0.0.1")
            port = int(parsed_opts.get("port", 8000)) if parsed_opts.get("port") else 8000
            token = parsed_opts.get("token")
            parsed_args = SimpleNamespace(host=host, port=port, token=token, bot_command="daemon")
        
        else:
            print(f"Error: unknown command: {cmd}", file=sys.stderr)
            sys.exit(1)
        
        return parsed_args
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nUsage:")
        print("  register url <url> [token <token>] - Register webhook with Telegram")
        print("  delete [token <token>] - Delete webhook from Telegram")
        print("  run [host <host>] [port <port>] [token <token>] [debug-rag] - Run server in foreground")
        print("  daemon [host <host>] [port <port>] [token <token>] - Run server as daemon")
        sys.exit(1)


def _execute_bot_command(parsed_args):
    """
    Execute bot command based on parsed arguments.
    
    Args:
        parsed_args: SimpleNamespace with parsed arguments including bot_command attribute
        
    Exits:
        If token is missing or on execution errors
    """
    # Get token from args or env
    token = getattr(parsed_args, 'token', None) or os.getenv("TELEGRAM_BOT_TOKEN")
    
    if parsed_args.bot_command in ["register", "delete", "run", "daemon"] and not token:
        print("Error: TELEGRAM_BOT_TOKEN must be set in environment or passed via --token", file=sys.stderr)
        sys.exit(1)
    
    # Set token in environment for app to use
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    
    # Execute command
    if parsed_args.bot_command == "register":
        register_webhook(parsed_args.url, token)
    elif parsed_args.bot_command == "delete":
        delete_webhook(token)
    elif parsed_args.bot_command == "run":
        log_level = getattr(parsed_args, 'log_level', None)
        debug_rag = getattr(parsed_args, 'debug_rag', False)
        run_server(parsed_args.host, parsed_args.port, log_level=log_level, debug_rag=debug_rag, args=parsed_args)
    elif parsed_args.bot_command == "daemon":
        run_daemon(parsed_args.host, parsed_args.port, args=parsed_args)


def main():
    """
    Main CLI entry point.
    """
    parsed_args = _parse_cli_arguments()
    _execute_bot_command(parsed_args)


if __name__ == "__main__":
    main()
