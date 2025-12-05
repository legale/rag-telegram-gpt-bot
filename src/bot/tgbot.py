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
from typing import Optional
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

# Load environment variables
load_dotenv()

# Global bot instance (loaded once)
bot_instance: Optional[LegaleBot] = None
telegram_app: Optional[Application] = None

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
    global bot_instance, telegram_app
    
    logger.info("Starting Legale Bot daemon...")
    
    # Load bot instance
    try:
        bot_instance = LegaleBot()
        logger.info("Bot core loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load bot core: {e}")
        raise
    
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
        
        # Handle message
        if update.message and update.message.text:
            await handle_message(update)
        
        return Response(status_code=200)
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return Response(status_code=500)


async def handle_message(update: Update):
    """
    Process incoming text messages.
    """
    message = update.message
    text = message.text
    chat_id = message.chat_id
    
    logger.info(f"Message from {chat_id}: {text[:50]}...")
    
    # Handle commands
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
            "Примеры вопросов:\n"
            "• Что случилось с точкой 840?\n"
            "• Когда Ru уходит в отпуск?\n"
            "• Какие были проблемы с сетью?\n\n"
            "Просто напишите свой вопрос!"
        )
    else:
        # Query the bot
        try:
            response = bot_instance.chat(text)
        except Exception as e:
            logger.error(f"Error querying bot: {e}", exc_info=True)
            response = "Извините, произошла ошибка при обработке вашего запроса."
    
    # Send response
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
