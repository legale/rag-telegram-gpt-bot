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
import argparse
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
from src.bot.admin_commands import ProfileCommands, HelpCommands, IngestCommands, StatsCommands, ControlCommands, SettingsCommands
from src.bot.admin_tasks import TaskManager
from src.bot.utils import AccessControlService, FrequencyController

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
            logger.error(f"Error resetting context: {e}", exc_info=True)
            return "Ошибка при сбросе контекста."
    
    async def handle_tokens_command(self) -> str:
        """Handle /tokens command."""
        try:
            usage = self.bot.get_token_usage()
            response = (
                f"📊 Использование токенов:\n\n"
                f"Текущее: {usage['current_tokens']:,}\n"
                f"Максимум: {usage['max_tokens']:,}\n"
                f"Использовано: {usage['percentage']}%\n\n"
            )
            if usage["percentage"] > 80:
                response += "⚠️ Приближаетесь к лимиту! Используйте /reset для сброса."
            elif usage["percentage"] > 50:
                response += "ℹ️ Контекст заполнен наполовину."
            else:
                response += "✅ Достаточно места для разговора."
            return response
        except Exception as e:
            logger.error(f"Error getting token usage: {e}", exc_info=True)
            return "Ошибка при получении информации о токенах."
    
    async def handle_model_command(self) -> str:
        """Handle /model command."""
        try:
            return self.bot.switch_model()
        except Exception as e:
            logger.error(f"Error switching model: {e}", exc_info=True)
            return "Ошибка при переключении модели."
    
    async def handle_admin_set_command(self, text: str, message) -> str:
        """Handle /admin_set command."""
        if not self.admin_manager:
            return "❌ Система администрирования недоступна. Установите ADMIN_PASSWORD в .env файле."
        
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return (
                "❌ Неверный формат команды.\n\n"
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
                logger.info(f"Admin set: {full_name} (ID: {user_id})")
                return (
                    f"✅ Вы успешно назначены администратором!\n\n"
                    f"👤 Имя: {full_name}\n"
                    f"🆔 ID: {user_id}\n"
                    f"📝 Username: @{username}"
                )
            except Exception as e:
                logger.error(f"Error setting admin: {e}", exc_info=True)
                return "❌ Ошибка при назначении администратора."
        else:
            logger.warning(f"Failed admin_set attempt from user {message.from_user.id}")
            return "❌ Неверный пароль."
    
    async def handle_admin_get_command(self, user_id: int) -> str:
        """Handle /admin_get command."""
        if not self.admin_manager:
            return "❌ Система администрирования недоступна."
        
        if not self.admin_manager.is_admin(user_id):
            logger.warning(f"Unauthorized admin_get attempt from user {user_id}")
            return "❌ Эта команда доступна только администратору."
        
        admin_info = self.admin_manager.get_admin()
        if admin_info:
            return (
                f"👤 Администратор бота:\n\n"
                f"Имя: {admin_info['full_name']}\n"
                f"ID: {admin_info['user_id']}\n"
                f"Username: @{admin_info['username']}"
            )
        else:
            return "❌ Администратор не назначен."
    
    async def handle_admin_command(self, update: Update) -> str:
        """Handle /admin command."""
        if not self.admin_router:
            return "❌ Админ-панель недоступна. Проверьте конфигурацию бота."
        
        try:
            return await self.admin_router.route(update, None, self.admin_manager)
        except Exception as e:
            logger.error(f"Error processing admin command: {e}", exc_info=True)
            return f"❌ Ошибка при выполнении админ-команды: {e}"
    
    async def handle_user_query(self, text: str, respond: bool) -> str:
        """Handle regular user query to bot."""
        try:
            return self.bot.chat(text, respond=respond)
        except Exception as e:
            logger.error(f"Error querying bot: {e}", exc_info=True)
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

    # пересоздаем core
    bot_instance = LegaleBot(
        db_url=paths["db_url"],
        vector_db_path=str(paths["vector_db_path"])
    )
    logger.warning(
        "Bot core (LegaleBot) initialized with profile=%s db_url=%s vector=%s",
        profile_manager.get_current_profile(),
        paths["db_url"],
        paths["vector_db_path"],
    )

    # пересоздаем admin_manager
    profile_dir = paths["profile_dir"]
    admin_manager_local = AdminManager(profile_dir)
    logger.info("Admin manager initialized for profile_dir=%s", profile_dir)

    # пересоздаем admin_router и все команды
    admin_router_local = AdminCommandRouter()
    task_manager_local = TaskManager()

    # profile commands
    profile_commands = ProfileCommands(profile_manager)
    admin_router_local.register("profile", profile_commands.list_profiles, "list")
    admin_router_local.register("profile", profile_commands.create_profile, "create")
    admin_router_local.register("profile", profile_commands.switch_profile, "switch")
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
    admin_router_local.register("chat", settings_commands.manage_chats)
    admin_router_local.register("allowed", settings_commands.manage_chats)
    admin_router_local.register("frequency", settings_commands.manage_frequency)

    # help commands
    help_commands = HelpCommands()
    admin_router_local.register("help", help_commands.show_help)

    logger.info("Admin router initialized with all admin commands")

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
    logger.warning("Hot reload requested for active profile...")
    paths = await init_runtime_for_current_profile()
    logger.warning(
        "Hot reload completed: profile=%s db=%s vector=%s",
        profile_manager.get_current_profile(),
        paths["db_path"],
        paths["vector_db_path"],
    )
    return paths


def setup_logging(verbosity: int = 0, use_syslog: bool = False):
    """
    Configure logging based on verbosity level.
    
    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG, 3=TRACE
        use_syslog: If True, log to syslog instead of stdout
    """
    levels = [logging.WARNING, logging.INFO, logging.DEBUG, logging.DEBUG]
    level = levels[min(verbosity, 3)]
    
    if use_syslog:
        from logging.handlers import SysLogHandler
        handler = SysLogHandler(address='/dev/log')
        formatter = logging.Formatter('legale-bot[%(process)d]: %(levelname)s - %(message)s')
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    
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
    
    logger.info("Starting Legale Bot daemon...")
    
    # Initialize profile manager
    try:
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        
        # Import ProfileManager from legale.py
        import sys
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from legale import ProfileManager
        
        profile_manager = ProfileManager(project_root)
        logger.info("Profile manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize profile manager: {e}")
        profile_manager = None
        raise RuntimeError("Profile manager initialization failed")

    # инициализация рантайма под активный профиль
    try:
        await init_runtime_for_current_profile()
    except Exception as e:
        logger.error(f"Failed to initialize runtime: {e}")
        raise
    
    # Initialize Telegram application
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment")
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    telegram_app = Application.builder().token(token).build()
    await telegram_app.initialize()
    logger.info("Telegram application initialized")
    
    # Initialize access control service
    if admin_manager:
        access_control = AccessControlService(admin_manager)
        logger.info("Access control service initialized")
    else:
        logger.warning("Access control not initialized - admin_manager is None")
    
    yield
    
    # Cleanup
    logger.info("Shutting down gracefully...")
    if telegram_app:
        await telegram_app.shutdown()
    logger.info("Shutdown complete")


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
        
        logger.debug(f"Received update: {update.update_id}")
        
        # Handle file upload (for ingestion)
        if update.message and update.message.document and ingest_commands:
            response_text = await ingest_commands.handle_file_upload(update, None, admin_manager)
            if response_text:
                await telegram_app.bot.send_message(chat_id=update.message.chat_id, text=response_text)
            return Response(status_code=200)
        
        # Handle message
        if update.message and update.message.text:
            await handle_message(update)
        
        return Response(status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
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

    logger.info(f"Message from {chat_id} (User {user_id}): {text[:50]}...")

    is_command = text.startswith("/")
    is_private = (message.chat.type == "private")

    # /id always responds, bypassing other logic
    if text == "/id":
        await telegram_app.bot.send_message(
            chat_id=chat_id,
            text=f"Chat ID: `{chat_id}`\nUser ID: `{user_id}`",
            parse_mode="Markdown",
        )
        return

    # admin_manager is required
    if not admin_manager:
        logger.error("admin_manager is not initialized, dropping message")
        return

    # Check access using AccessControlService
    if not access_control:
        logger.error("access_control is not initialized")
        return
    
    is_allowed, denial_reason = access_control.is_allowed(
        user_id=user_id,
        chat_id=chat_id,
        is_private=is_private,
        is_command=is_command
    )
    
    if not is_allowed:
        # Access denied - silently ignore or log
        return

    # Determine if bot should respond using FrequencyController
    config = admin_manager.config
    bot_username = (telegram_app.bot.username or "").lower()
    bot_id = telegram_app.bot.id
    has_mention = is_bot_mentioned(message, bot_username, bot_id)
    
    respond, reason = frequency_controller.should_respond(
        chat_id=chat_id,
        frequency=config.response_frequency or 0,
        has_mention=has_mention,
        is_command=is_command,
        is_private=is_private
    )

    # Route message to appropriate handler
    handler = MessageHandler(bot_instance, admin_manager, admin_router)
    
    if is_command:
        response = await handler.route_command(text, update)
        if response is None:
            # Not a recognized command, treat as regular query
            response = await handler.handle_user_query(text, respond)
    else:
        # Regular user query
        response = await handler.handle_user_query(text, respond)

    # Send response if available
    if response:
        await telegram_app.bot.send_message(chat_id=chat_id, text=response)
        logger.info(f"Response sent to {chat_id}")


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
            print(f"✓ Webhook registered successfully: {url}")
            print(f"  Description: {result.get('description', 'N/A')}")
        else:
            print(f"✗ Failed to register webhook: {result.get('description')}")
            sys.exit(1)
    else:
        print(f"✗ HTTP error: {response.status_code}")
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
            print("✓ Webhook deleted successfully")
        else:
            print(f"✗ Failed to delete webhook: {result.get('description')}")
            sys.exit(1)
    else:
        print(f"✗ HTTP error: {response.status_code}")
        sys.exit(1)


def run_server(host: str = "127.0.0.1", port: int = 8000, verbosity: int = 0):
    """
    Run the FastAPI server in foreground mode.
    """
    setup_logging(verbosity=verbosity, use_syslog=False)
    
    log_level = ["warning", "info", "debug", "trace"][min(verbosity, 3)]
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=verbosity >= 2
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
        logger.info("Daemon started")
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
    parser = argparse.ArgumentParser(
        description="Legale Bot Telegram Webhook Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register webhook
    register_parser = subparsers.add_parser("register", help="Register webhook with Telegram")
    register_parser.add_argument("--url", required=True, help="Webhook URL (e.g., https://example.com/webhook)")
    register_parser.add_argument("--token", help="Bot token (or set TELEGRAM_BOT_TOKEN env var)")
    
    # Delete webhook
    delete_parser = subparsers.add_parser("delete", help="Delete webhook from Telegram")
    delete_parser.add_argument("--token", help="Bot token (or set TELEGRAM_BOT_TOKEN env var)")
    
    # Run foreground
    run_parser = subparsers.add_parser("run", help="Run server in foreground")
    run_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    run_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    run_parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level (-v, -vv, -vvv)")
    run_parser.add_argument("--token", help="Bot token (or set TELEGRAM_BOT_TOKEN env var)")
    
    # Run daemon
    daemon_parser = subparsers.add_parser("daemon", help="Run server as daemon")
    daemon_parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    daemon_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    daemon_parser.add_argument("--token", help="Bot token (or set TELEGRAM_BOT_TOKEN env var)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Get token from args or env
    token = args.token if hasattr(args, 'token') and args.token else os.getenv("TELEGRAM_BOT_TOKEN")
    
    if args.command in ["register", "delete", "run", "daemon"] and not token:
        print("Error: TELEGRAM_BOT_TOKEN must be set in environment or passed via --token")
        sys.exit(1)
    
    # Set token in environment for app to use
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    
    # Execute command
    if args.command == "register":
        register_webhook(args.url, token)
    elif args.command == "delete":
        delete_webhook(token)
    elif args.command == "run":
        run_server(args.host, args.port, args.verbose)
    elif args.command == "daemon":
        run_daemon(args.host, args.port)


if __name__ == "__main__":
    main()
