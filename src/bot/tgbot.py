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

# Logging setup
logger = logging.getLogger("legale_tgbot")


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
    global bot_instance, telegram_app, admin_manager, admin_router, profile_manager
    
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

    # Load bot instance with profile paths
    try:
        # Get profile paths
        paths = profile_manager.get_profile_paths()
        
        bot_instance = LegaleBot(
            db_url=paths['db_url'],
            vector_db_path=str(paths['vector_db_path'])
        )
        logger.info(f"Bot core loaded successfully with profile: {profile_manager.get_current_profile()}")
        logger.info(f"DB: {paths['db_url']}")
    except Exception as e:
        logger.error(f"Failed to load bot core: {e}")
        raise
    
    # Initialize admin manager
    try:
        # Get profile directory
        profile_dir = paths['profile_dir']
        admin_manager = AdminManager(profile_dir)
        logger.info(f"Admin manager initialized for profile: {profile_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize admin manager: {e}")
        admin_manager = None
    
    # Initialize admin router
    if admin_manager and profile_manager:
        try:
            admin_router = AdminCommandRouter()
            task_manager = TaskManager()
            
            # Register profile commands
            profile_commands = ProfileCommands(profile_manager)
            admin_router.register('profile', profile_commands.list_profiles, 'list')
            admin_router.register('profile', profile_commands.create_profile, 'create')
            admin_router.register('profile', profile_commands.switch_profile, 'switch')
            admin_router.register('profile', profile_commands.delete_profile, 'delete')
            admin_router.register('profile', profile_commands.profile_info, 'info')
            
            # Register ingest commands
            ingest_commands = IngestCommands(profile_manager, task_manager)
            admin_router.register('ingest', ingest_commands.start_ingest)
            admin_router.register('ingest', ingest_commands.clear_data, 'clear')
            admin_router.register('ingest', ingest_commands.ingest_status, 'status')
            
            # Register stats commands
            stats_commands = StatsCommands(profile_manager)
            admin_router.register('stats', stats_commands.show_stats)
            admin_router.register('health', stats_commands.health_check)
            admin_router.register('logs', stats_commands.show_logs)
            
            # Register control commands
            control_commands = ControlCommands(profile_manager)
            admin_router.register('restart', control_commands.restart_bot)
            
            # Register settings commands
            settings_commands = SettingsCommands(profile_manager)
            admin_router.register('chat', settings_commands.manage_chats)
            admin_router.register('frequency', settings_commands.manage_frequency)
            
            # Register help commands
            help_commands = HelpCommands()
            admin_router.register('help', help_commands.show_help)
            
            logger.info("Admin router initialized with profile, ingest, stats, control, settings, and help commands")
        except Exception as e:
            logger.error(f"Failed to initialize admin router: {e}")
            admin_router = None
    else:
        logger.warning("Admin router not initialized (missing admin_manager or profile_manager)")
        admin_router = None
    
    # Initialize Telegram application
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment")
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    telegram_app = Application.builder().token(token).build()
    await telegram_app.initialize()
    logger.info("Telegram application initialized")
    
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

async def handle_message(update: Update):
    """
    Process incoming text messages.
    """
    message = update.message
    text = message.text
    chat_id = message.chat_id
    user_id = message.from_user.id
    
    logger.info(f"Message from {chat_id} (User {user_id}): {text[:50]}...")
    
    # 1. Check commands
    is_command = text.startswith('/')
    
    # Always allow admin commands for setup
    if text.startswith('/admin'):
        # Pass through to admin handlers logic below
        pass
    elif text == '/id':
        await telegram_app.bot.send_message(chat_id=chat_id, text=f"Chat ID: `{chat_id}`\nUser ID: `{user_id}`", parse_mode="Markdown")
        return
    else:
        # Check whitelist
        config = admin_manager.config
        if chat_id not in config.allowed_chats:
            # Ignore unauthorized chats
            logger.info(f"Ignoring message from unauthorized chat {chat_id}")
            return
    
    # Check frequency (only for non-commands)
    respond = True
    if not is_command:
        config = admin_manager.config
        freq = config.response_frequency
        
        if freq > 1:
            current = chat_counters.get(chat_id, 0)
            current += 1
            chat_counters[chat_id] = current
            
            # Respond every Nth message (1st, N+1, 2N+1...)
            # Actually user asked: "if 3, respond only to every 3rd".
            # Usually implies: 1(no), 2(no), 3(yes).
            if current % freq != 0:
                respond = False
                logger.debug(f"Skipping response due to frequency (msg {current}, freq {freq})")
            else:
                logger.debug(f"Responding due to frequency (msg {current}, freq {freq})")

    # Handle commands logic (existing)
    if text.startswith('/start'):
        response = (
            "Привет! Я Legale Bot — юрист профсоюза IT-работников.\n\n"
            "Задавайте вопросы о ваших правах, рабочих ситуациях, "
            "и я постараюсь помочь на основе истории чата.\n\n"
            "Используйте /help для справки."
        )
    elif text.startswith('/help'):
        response = (
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
            "Примеры вопросов:\n"
            "• Что случилось с точкой 840?\n"
            "• Когда Ru уходит в отпуск?\n"
            "• Какие были проблемы с сетью?\n\n"
            "Просто напишите свой вопрос!"
        )
    elif text.startswith('/reset'):
        try:
            response = bot_instance.reset_context()
        except Exception as e:
            logger.error(f"Error resetting context: {e}", exc_info=True)
            response = "Ошибка при сбросе контекста."
    elif text.startswith('/tokens'):
        try:
            usage = bot_instance.get_token_usage()
            response = (
                f"📊 Использование токенов:\n\n"
                f"Текущее: {usage['current_tokens']:,}\n"
                f"Максимум: {usage['max_tokens']:,}\n"
                f"Использовано: {usage['percentage']}%\n\n"
            )
            if usage['percentage'] > 80:
                response += "⚠️ Приближаетесь к лимиту! Используйте /reset для сброса."
            elif usage['percentage'] > 50:
                response += "ℹ️ Контекст заполнен наполовину."
            else:
                response += "✅ Достаточно места для разговора."
        except Exception as e:
            logger.error(f"Error getting token usage: {e}", exc_info=True)
            response = "Ошибка при получении информации о токенах."
    elif text.startswith('/model'):
        try:
            response = bot_instance.switch_model()
        except Exception as e:
            logger.error(f"Error switching model: {e}", exc_info=True)
            response = "Ошибка при переключении модели."
    
    elif text.startswith('/admin_set'):
        # Command format: /admin_set PASSWORD
        if not admin_manager:
            response = "❌ Система администрирования недоступна. Установите ADMIN_PASSWORD в .env файле."
        else:
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                response = (
                    "❌ Неверный формат команды.\\n\\n"
                    "Использование: /admin_set <пароль>\\n\\n"
                    "Пример: /admin_set my_secret_password"
                )
            else:
                password = parts[1].strip()
                
                if admin_manager.verify_password(password):
                    # Get user info from message
                    user = message.from_user
                    user_id = user.id
                    username = user.username or "unknown"
                    first_name = user.first_name or "Unknown"
                    last_name = user.last_name
                    
                    try:
                        admin_manager.set_admin(user_id, username, first_name, last_name)
                        full_name = f"{first_name} {last_name}".strip() if last_name else first_name
                        response = (
                            f"✅ Вы успешно назначены администратором!\\n\\n"
                            f"👤 Имя: {full_name}\\n"
                            f"🆔 ID: {user_id}\\n"
                            f"📝 Username: @{username}"
                        )
                        logger.info(f"Admin set: {full_name} (ID: {user_id})")
                    except Exception as e:
                        logger.error(f"Error setting admin: {e}", exc_info=True)
                        response = "❌ Ошибка при назначении администратора."
                else:
                    response = "❌ Неверный пароль."
                    logger.warning(f"Failed admin_set attempt from user {message.from_user.id}")
    
    elif text.startswith('/admin_get'):
        if not admin_manager:
            response = "❌ Система администрирования недоступна."
        else:
            # Check if requester is admin
            requester_id = message.from_user.id
            
            if not admin_manager.is_admin(requester_id):
                response = "❌ Эта команда доступна только администратору."
                logger.warning(f"Unauthorized admin_get attempt from user {requester_id}")
            else:
                admin_info = admin_manager.get_admin()
                if admin_info:
                    response = (
                        f"👤 Администратор бота:\\n\\n"
                        f"Имя: {admin_info['full_name']}\\n"
                        f"ID: {admin_info['user_id']}\\n"
                        f"Username: @{admin_info['username']}"
                    )
                else:
                    response = "❌ Администратор не назначен."
    
    elif text.startswith('/admin'):
        # Admin commands
        if not admin_router:
            response = "❌ Админ-панель недоступна. Проверьте конфигурацию бота."
        else:
            try:
                response = await admin_router.route(update, None, admin_manager)
            except Exception as e:
                logger.error(f"Error processing admin command: {e}", exc_info=True)
                response = f"❌ Ошибка при выполнении админ-команды: {e}"
    
    else:
        # Query the bot
        try:
            response = bot_instance.chat(text, respond=respond)
        except Exception as e:
            logger.error(f"Error querying bot: {e}", exc_info=True)
            response = f"Произошла ошибка при обработке вашего запроса. error={e}"
    
    # Send response
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
