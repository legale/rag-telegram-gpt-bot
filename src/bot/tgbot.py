#!/usr/bin/env python3
"""
Telegram Bot Webhook Daemon for Legale Bot.

This module provides:
- FastAPI webhook endpoint for Telegram updates
- CLI utilities for webhook registration/deletion
- Daemon and foreground modes with verbosity control
- Persistent in-memory LegaleBot instance
"""

import sys
import os
import logging
import signal
from typing import Optional, Dict, List
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
from src.bot.admin_commands import ProfileCommands, HelpCommands, IngestCommands, StatsCommands, ControlCommands, SettingsCommands, ModelCommands, SystemPromptCommands
from src.bot.admin_tasks import TaskManager
from src.bot.utils import AccessControlService, FrequencyController
from src.core.syslog2 import *

# Load environment variables
load_dotenv()

# Global bot instance (loaded once)
bot_instance: Optional[LegaleBot] = None
telegram_app: Optional[Application] = None
admin_manager: Optional[AdminManager] = None
admin_router: Optional[AdminCommandRouter] = None
profile_manager = None  # Will be initialized in lifespan
task_manager: Optional[TaskManager] = None
ingest_commands: Optional[IngestCommands] = None

# Access control and frequency controller
access_control: Optional[AccessControlService] = None
frequency_controller: FrequencyController = FrequencyController()

# Debug RAG mode flag
debug_rag_mode: bool = False

# Logging setup
logger = logging.getLogger("legale_tgbot")

# Chat message counters for frequency control (deprecated - now in FrequencyController)
chat_counters: Dict[int, int] = {}


class MessageHandler:
    """Handles message routing and command processing."""
    
    def __init__(self, bot_instance, admin_manager, admin_router):
        self.bot = bot_instance
        self.admin_manager = admin_manager
        self.admin_router = admin_router
    
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
            syslog2(LOG_ERR, "reset context failed", error=str(e))
            return "Ошибка при сбросе контекста."
    
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
            syslog2(LOG_ERR, "get token usage failed", error=str(e))
            return "Ошибка при получении информации о токенах."
    
    async def handle_model_command(self) -> str:
        """Handle /model command."""
        try:
            msg = self.bot.get_model()
            # Save new model to config
            if self.admin_manager:
                self.admin_manager.config.current_model = self.bot.current_model_name
            return msg
        except Exception as e:
            syslog2(LOG_ERR, "get model failed", error=str(e))
            return "Ошибка при переключении модели."
    
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
                syslog2(LOG_ERR, "set admin failed", error=str(e))
                return "Ошибка при назначении администратора."
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
            syslog2(LOG_ERR, "admin command failed", error=str(e))
            return f"Ошибка при выполнении админ-команды: {e}"
    
    async def handle_user_query(self, text: str, respond: bool) -> str:
        """Handle regular user query to bot."""
        try:
            # Get system prompt from config
            system_prompt_template = self.admin_manager.config.system_prompt
            
            # Debug RAG mode - show retrieved chunks and prompts
            if debug_rag_mode and respond:
                debug_info = self.bot.get_rag_debug_info(text, n_results=3)
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
            
            return self.bot.chat(text, respond=respond, system_prompt_template=system_prompt_template)
        except Exception as e:
            syslog2(LOG_ERR, "process user query failed", error=str(e))
            return f"Произошла ошибка при обработке вашего запроса. error={e}"
    
    async def route_command(self, text: str, update: Update) -> str:
        """Route command to appropriate handler."""
        message = update.message
        user_id = message.from_user.id
        
        if text.startswith("/start"):
            return await self.handle_start_command()
        elif text.startswith("/help"):
            return await self.handle_help_command()
        elif text.startswith("/reset"):
            return await self.handle_reset_command()
        elif text.startswith("/tokens"):
            return await self.handle_tokens_command()
        elif text.startswith("/model"):
            return await self.handle_model_command()
        elif text.startswith("/admin_set"):
            return await self.handle_admin_set_command(text, message)
        elif text.startswith("/admin_get"):
            return await self.handle_admin_get_command(user_id)
        elif text.startswith("/admin"):
            return await self.handle_admin_command(update)
        else:
            return None  # Not a recognized command


# инициализация рантайма под текущий профиль
async def init_runtime_for_current_profile():
    """
    создать/переинициализировать bot_instance, admin_manager, admin_router и связанные команды
    под текущий активный профиль profile_manager
    """
    global bot_instance, admin_manager, admin_router, task_manager, ingest_commands

    if profile_manager is None:
        raise RuntimeError("profile_manager is not initialized")

    # получаем пути для текущего профиля (использует ACTIVE_PROFILE из .env)
    paths = profile_manager.get_profile_paths()

    # пересоздаем admin_manager sначала, чтобы получить конфиг
    profile_dir = paths["profile_dir"]
    admin_manager_local = AdminManager(profile_dir)
    syslog2(LOG_NOTICE, "admin manager initialized", profile_dir=str(profile_dir))

    # Загружаем сохраненную модель
    model_name = admin_manager_local.config.current_model or "openai/gpt-oss-20b:free"

    # пересоздаем core
    bot_instance = LegaleBot(
        db_url=paths["db_url"],
        vector_db_path=str(paths["vector_db_path"]),
        model_name=model_name,
        profile_dir=profile_dir
    )
    syslog2(LOG_WARNING, "bot core initialized", profile=profile_manager.get_current_profile(), db_url=paths["db_url"], vector=paths["vector_db_path"], model=model_name)

    # пересоздаем admin_router и все команды
    admin_router_local = AdminCommandRouter()
    task_manager_local = TaskManager()

    # profile commands
    profile_commands = ProfileCommands(profile_manager)
    admin_router_local.register("profile", profile_commands.list_profiles, "list")
    admin_router_local.register("profile", profile_commands.create_profile, "create")
    admin_router_local.register("profile", profile_commands.get_profile, "get")
    admin_router_local.register("profile", profile_commands.set_profile, "set")
    admin_router_local.register("profile", profile_commands.delete_profile, "delete")
    admin_router_local.register("profile", profile_commands.profile_info, "info")

    # ingest commands
    ingest_commands_local = IngestCommands(profile_manager, task_manager_local)
    admin_router_local.register("ingest", ingest_commands_local.start_ingest)
    admin_router_local.register("ingest", ingest_commands_local.clear_data, "clear")
    admin_router_local.register("ingest", ingest_commands_local.ingest_status, "status")

    # stats commands
    stats_commands = StatsCommands(profile_manager)
    admin_router_local.register("stats", stats_commands.show_stats)
    admin_router_local.register("health", stats_commands.health_check)
    admin_router_local.register("logs", stats_commands.show_logs)

    # control commands – сюда прокидываем колбэк hot-reload
    control_commands = ControlCommands(profile_manager, reload_callback=reload_for_current_profile)
    admin_router_local.register("restart", control_commands.restart_bot)

    # settings commands
    settings_commands = SettingsCommands(profile_manager)
    admin_router_local.register("allowed", settings_commands.manage_chats, "list")
    admin_router_local.register("allowed", settings_commands.manage_chats, "add")
    admin_router_local.register("allowed", settings_commands.manage_chats, "remove")
    admin_router_local.register("chat", settings_commands.manage_chats)
    admin_router_local.register("frequency", settings_commands.manage_frequency)

    # model commands
    model_commands = ModelCommands(profile_manager, bot_instance)
    admin_router_local.register("model", model_commands.list_models, "list")
    admin_router_local.register("model", model_commands.get_model, "get")
    admin_router_local.register("model", model_commands.set_model, "set")

    # system prompt commands
    system_prompt_commands = SystemPromptCommands(profile_manager)
    admin_router_local.register("system_prompt", system_prompt_commands.get_prompt, "get")
    admin_router_local.register("system_prompt", system_prompt_commands.set_prompt, "set")
    admin_router_local.register("system_prompt", system_prompt_commands.reset_prompt, "reset")

    # help command
    help_commands = HelpCommands()
    admin_router_local.register("help", help_commands.show_help)

    syslog2(LOG_INFO, "admin router initialized")



    # только после успешного создания всех локальных объектов – публикуем их в глобальные
    admin_manager = admin_manager_local
    admin_router = admin_router_local
    task_manager = task_manager_local
    ingest_commands = ingest_commands_local

    return paths


# hot-reload рантайма под активный профиль (используется /admin restart)
async def reload_for_current_profile():
    """
    hot-reload рантайма под активный профиль (используется /admin restart)
    """
    syslog2(LOG_WARNING, "hot reload requested")
    paths = await init_runtime_for_current_profile()
    syslog2(LOG_WARNING, "hot reload completed", profile=profile_manager.get_current_profile(), db=paths["db_path"], vector=paths["vector_db_path"])
    return paths


def setup_logging(log_level: Optional[str] = None, verbosity: Optional[int] = None, use_syslog: bool = False):
    """
    Configure logging based on log level or verbosity.
    
    Args:
        log_level: Log level string (INFO, DEBUG, WARNING, etc.) or number (6=INFO, 7=DEBUG)
        verbosity: Legacy verbosity level (0=WARNING, 1=INFO, 2=DEBUG, 3=TRACE) - for backward compatibility
        use_syslog: If True, log to syslog instead of stdout
    """
    # Map log_level to logging level
    if log_level:
        log_level_upper = log_level.upper()
        level_map = {
            "1": logging.CRITICAL,  # ALERT
            "2": logging.CRITICAL,  # CRIT
            "3": logging.ERROR,     # ERR
            "4": logging.WARNING,   # WARNING
            "5": logging.INFO,      # NOTICE
            "6": logging.INFO,      # INFO
            "7": logging.DEBUG,     # DEBUG
            "ALERT": logging.CRITICAL,
            "CRIT": logging.CRITICAL,
            "ERR": logging.ERROR,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "NOTICE": logging.INFO,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }
        level = level_map.get(log_level_upper, logging.WARNING)
    elif verbosity is not None:
        # Legacy support for verbosity
        levels = [logging.WARNING, logging.INFO, logging.DEBUG, logging.DEBUG]
        level = levels[min(verbosity, 3)]
    else:
        level = logging.WARNING
    
    if use_syslog:
        from logging.handlers import SysLogHandler
        handler = SysLogHandler(address='/dev/log')
        formatter = logging.Formatter('legale-bot[%(process)d]: %(levelname)s - %(message)s')
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)
    # Configure root logger to capture all logs including syslog2 ("app")
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Also configure uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads bot instance on startup, cleans up on shutdown.
    """
    global bot_instance, telegram_app, admin_manager, admin_router, profile_manager, task_manager, ingest_commands, access_control
    
    syslog2(LOG_NOTICE, "daemon starting")
    
    # Initialize profile manager
    try:
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        
        # Import ProfileManager from legale.py
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from legale import ProfileManager
        
        # Create ProfileManager instance
        profile_manager = ProfileManager(project_root)
        
        logger.info("Profile manager initialized")
        syslog2(LOG_NOTICE, "profile manager initialized", profile=profile_manager.get_current_profile())
    except Exception as e:
        syslog2(LOG_ERR, "profile manager init failed", error=str(e))
        profile_manager = None
        raise RuntimeError("Profile manager initialization failed")

    # инициализация рантайма под активный профиль
    try:
        await init_runtime_for_current_profile()
    except Exception as e:
        syslog2(LOG_ERR, "runtime init failed", error=str(e))
        raise
    
    # Initialize Telegram application
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        syslog2(LOG_ERR, "telegram token missing")
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    telegram_app = Application.builder().token(token).build()
    await telegram_app.initialize()
    syslog2(LOG_NOTICE, "telegram app initialized")
    
    # Initialize access control service
    if admin_manager:
        access_control = AccessControlService(admin_manager)
        syslog2(LOG_NOTICE, "access control initialized")
    else:
        syslog2(LOG_WARNING, "access control not initialized", reason="admin_manager is None")
    
    yield
    
    # Cleanup
    syslog2(LOG_NOTICE, "shutting down")
    if telegram_app:
        await telegram_app.shutdown()
    syslog2(LOG_NOTICE, "shutdown complete")


# Create FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "bot_loaded": bot_instance is not None}


@app.post("/webhook")
async def webhook(request: Request):
    """
    Handle incoming Telegram webhook updates.
    """
    try:
        # Parse update
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        
        syslog2(LOG_DEBUG, "update received", update_id=update.update_id)
        
        # Handle file upload (for ingestion)
        if update.message and update.message.document and ingest_commands:
            response_text = await ingest_commands.handle_file_upload(update, None, admin_manager)
            if response_text:
                await telegram_app.bot.send_message(chat_id=update.message.chat_id, text=response_text)
            return Response(status_code=200)
        
        # Handle message
        if update.message and update.message.text:
            try:
                await handle_message(update)
            except Exception as e:
                syslog2(LOG_ERR, "handle_message failed", error=str(e), update_id=update.update_id)
                # Try to send error message to user
                try:
                    if update.message:
                        await telegram_app.bot.send_message(
                            chat_id=update.message.chat_id,
                            text="Произошла ошибка при обработке сообщения. Попробуйте позже."
                        )
                except:
                    pass
        
        return Response(status_code=200)
    
    except Exception as e:
        syslog2(LOG_ERR, "webhook processing failed", error=str(e))
        return Response(status_code=500)


# Chat message counters for frequency control
chat_counters: Dict[int, int] = {}


def is_bot_mentioned(message, bot_username: str, bot_id: int) -> bool:
    """
    check if bot is mentioned in message (by @username or text_mention)
    """
    text = message.text or ""
    entities = message.entities or []
    if not entities:
        return False

    for ent in entities:
        try:
            if ent.type == "mention":
                mention_text = text[ent.offset: ent.offset + ent.length]
                if bot_username and mention_text.lower() == f"@{bot_username}":
                    return True
            elif ent.type == "text_mention" and ent.user and ent.user.id == bot_id:
                return True
        except Exception:
            continue

    return False


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

    # Public commands that should always respond, bypassing access control
    # These commands are available to everyone, including non-admins
    public_commands = ["/id", "/help", "/admin_set"]
    
    # Handle /id command
    if text == "/id":
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text=f"Chat ID: `{chat_id}`\nUser ID: `{user_id}`",
            parse_mode="Markdown",
        )
        return
    
    # Handle /help command - need admin_manager for handler
    if text == "/help" or (is_command and text.startswith("/help")):
        if not admin_manager:
            syslog2(LOG_ERR, "admin manager missing", action="drop_message")
            return
        
        # Create handler and process /help
        if not bot_instance:
            syslog2(LOG_ERR, "bot instance missing", action="drop_message")
            return
        
        handler = MessageHandler(bot_instance, admin_manager, admin_router)
        response = await handler.handle_help_command()
        if response:
            await telegram_app.bot.send_message(chat_id=chat_id, text=response)
        return
    
    # Handle /admin_set command - must be available to everyone to set themselves as admin
    if is_command and text.startswith("/admin_set"):
        if not admin_manager:
            syslog2(LOG_ERR, "admin manager missing", action="drop_message")
            return
        
        if not bot_instance:
            syslog2(LOG_ERR, "bot instance missing", action="drop_message")
            return
        
        handler = MessageHandler(bot_instance, admin_manager, admin_router)
        response = await handler.handle_admin_set_command(text, message)
        if response:
            await telegram_app.bot.send_message(chat_id=chat_id, text=response)
        return

    # admin_manager is required for other commands
    if not admin_manager:
        syslog2(LOG_ERR, "admin manager missing", action="drop_message")
        return

    # Check access using AccessControlService
    if not access_control:
        syslog2(LOG_ERR, "access control missing")
        return
    
    is_allowed, denial_reason = access_control.is_allowed(
        user_id=user_id,
        chat_id=chat_id,
        is_private=is_private,
        is_command=is_command
    )
    
    if not is_allowed:
        # Access denied - log reason
        syslog2(LOG_DEBUG, "access denied", chat_id=chat_id, user_id=user_id, reason=denial_reason)
        return

    # Determine if bot should respond using FrequencyController
    config = admin_manager.config
    bot_username = (telegram_app.bot.username or "").lower()
    bot_id = telegram_app.bot.id
    has_mention = is_bot_mentioned(message, bot_username, bot_id)
    
    # Parse search command
    is_search_command = False
    search_query = ""
    if has_mention and not is_command:
        raw_text = (text or "").strip()
        lowered = raw_text.lower()
        mention_prefix = f"@{bot_username}"
        if lowered.startswith(mention_prefix):
            raw_text = raw_text[len(mention_prefix):].lstrip()
            parts = raw_text.split(maxsplit=1)
            if parts:
                first = parts[0].lower()
                rest = parts[1].strip() if len(parts) > 1 else ""
                if first in ("поиск", "find") and rest:
                    is_search_command = True
                    search_query = rest
    
    respond, reason = frequency_controller.should_respond(
        chat_id=chat_id,
        frequency=config.response_frequency or 0,
        has_mention=has_mention,
        is_command=is_command,
        is_private=is_private
    )
    
    syslog2(LOG_DEBUG, "response decision", chat_id=chat_id, respond=respond, reason=reason, is_command=is_command, is_private=is_private, has_mention=has_mention)

    # Check if bot_instance is available
    if not bot_instance:
        syslog2(LOG_ERR, "bot instance missing", action="drop_message")
        return
    
    # Handle search command
    if is_search_command:
        db = bot_instance.db
        retrieval = bot_instance.retrieval
        from src.core.message_search import search_message_links
        
        links = search_message_links(retrieval, db, search_query, top_k=3)
        
        if not links:
            reply = f'по запросу "{search_query}" ничего не найдено'
        else:
            lines = [f'найдено по запросу: "{search_query}"']
            for idx, link in enumerate(links, start=1):
                lines.append(f"{idx}. {link}")
            reply = "\n".join(lines)
        
        try:
            await telegram_app.bot.send_message(chat_id=chat_id, text=reply)
            syslog2(LOG_NOTICE, "search response sent", chat_id=chat_id, query=search_query, links=len(links))
        except Exception as e:
            syslog2(LOG_ERR, "failed to send search response", chat_id=chat_id, error=str(e))
        return
    
    # Route message to appropriate handler
    handler = MessageHandler(bot_instance, admin_manager, admin_router)
    
    response = None
    if is_command:
        response = await handler.route_command(text, update)
        if response is None:
            # Not a recognized command, treat as regular query
            response = await handler.handle_user_query(text, respond)
    else:
        # Regular user query
        # Если частота 0 и мы не отвечаем (нет упоминания), то полностью игнорируем сообщение
        # (не сохраняем в историю и не тратим ресурсы)
        target_freq = config.response_frequency or 0
        if not respond and target_freq == 0:
            syslog2(LOG_DEBUG, "message ignored", chat_id=chat_id, reason="freq=0, no mention")
            return

        response = await handler.handle_user_query(text, respond)

    # Send response if available
    if response:
        try:
            await telegram_app.bot.send_message(chat_id=chat_id, text=response)
            syslog2(LOG_NOTICE, "response sent", chat_id=chat_id, response_length=len(response))
        except Exception as e:
            syslog2(LOG_ERR, "failed to send response", chat_id=chat_id, error=str(e))
    else:
        syslog2(LOG_DEBUG, "no response generated", chat_id=chat_id, is_command=is_command, respond=respond)


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


def run_server(host: str = "127.0.0.1", port: int = 8000, log_level: Optional[str] = None, debug_rag: bool = False):
    """
    Run the FastAPI server in foreground mode.
    
    Args:
        host: Host to bind
        port: Port to bind
        log_level: Log level string (INFO, DEBUG, WARNING, etc.) or number (6=INFO, 7=DEBUG)
        debug_rag: Enable RAG debug mode
    """
    global debug_rag_mode
    debug_rag_mode = debug_rag
    
    setup_logging(log_level=log_level, use_syslog=False)
    
    # Map log_level to uvicorn log level (lowercase string)
    if log_level:
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
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        access_log=access_log
    )


def run_daemon(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the FastAPI server in daemon mode (background).
    """
    import daemon
    from daemon import pidfile
    
    pid_file = "/var/run/legale-bot.pid"
    
    # Setup syslog logging
    setup_logging(verbosity=1, use_syslog=True)
    
    with daemon.DaemonContext(
        pidfile=pidfile.TimeoutPIDLockFile(pid_file),
        signal_map={
            signal.SIGTERM: lambda signum, frame: sys.exit(0),
            signal.SIGINT: lambda signum, frame: sys.exit(0),
        }
    ):
        syslog2(LOG_NOTICE, "daemon started")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )


def main():
    """
    Main CLI entry point.
    """
    from src.core.cli_parser import (
        CommandParser, CommandSpec, ArgStream, CLIError, CLIHelp,
        parse_option, parse_int_option, parse_flag
    )
    
    def parse_bot_register(stream: ArgStream) -> dict:
        """Parse bot register command."""
        url = parse_option(stream, "url")
        if not url:
            raise CLIError("url required for bot register")
        token = parse_option(stream, "token")
        return {"url": url, "token": token, "bot_command": "register"}
    
    def parse_bot_delete(stream: ArgStream) -> dict:
        """Parse bot delete command."""
        token = parse_option(stream, "token")
        return {"token": token, "bot_command": "delete"}
    
    def parse_bot_run(stream: ArgStream) -> dict:
        """Parse bot run command."""
        host = parse_option(stream, "host") or "127.0.0.1"
        port = parse_int_option(stream, "port", 8000)
        token = parse_option(stream, "token")
        debug_rag = parse_flag(stream, "debug-rag")
        return {"host": host, "port": port, "token": token, "debug_rag": debug_rag, "bot_command": "run"}
    
    def parse_bot_daemon(stream: ArgStream) -> dict:
        """Parse bot daemon command."""
        host = parse_option(stream, "host") or "127.0.0.1"
        port = parse_int_option(stream, "port", 8000)
        token = parse_option(stream, "token")
        return {"host": host, "port": port, "token": token, "bot_command": "daemon"}
    
    commands = [
        CommandSpec("register", parse_bot_register, "Register webhook with Telegram\n  register url <url> [token <token>]"),
        CommandSpec("delete", parse_bot_delete, "Delete webhook from Telegram\n  delete [token <token>]"),
        CommandSpec("run", parse_bot_run, "Run server in foreground\n  run [host <host>] [port <port>] [token <token>] [debug-rag] [-V <level>]"),
        CommandSpec("daemon", parse_bot_daemon, "Run server as daemon\n  daemon [host <host>] [port <port>] [token <token>]"),
    ]
    
    parser = CommandParser(commands)
    
    try:
        cmd_name, args = parser.parse(sys.argv[1:])
    except CLIHelp:
        print("Legale Bot Telegram Webhook Daemon")
        print("\nCommands:")
        for spec in commands:
            if spec.help_text:
                print(f"  {spec.help_text}")
        sys.exit(0)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get token from args or env
    token = getattr(args, 'token', None) or os.getenv("TELEGRAM_BOT_TOKEN")
    
    if cmd_name in ["register", "delete", "run", "daemon"] and not token:
        print("Error: TELEGRAM_BOT_TOKEN must be set in environment or passed via --token", file=sys.stderr)
        sys.exit(1)
    
    # Set token in environment for app to use
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    
    # Execute command
    if cmd_name == "register":
        register_webhook(args.url, token)
    elif cmd_name == "delete":
        delete_webhook(token)
    elif cmd_name == "run":
        log_level = getattr(args, 'log_level', None)
        debug_rag = getattr(args, 'debug_rag', False)
        run_server(args.host, args.port, log_level=log_level, debug_rag=debug_rag)
    elif cmd_name == "daemon":
        run_daemon(args.host, args.port)


if __name__ == "__main__":
    main()
