#!/usr/bin/env python3
"""
Legale Bot - Unified CLI Orchestrator

This is the main entry point for all Legale Bot operations.
It provides a unified interface for managing profiles, ingesting data,
running the bot, and managing the Telegram webhook.
"""

import os
import sys
import subprocess
import shutil
import asyncio
from pathlib import Path

# Check if we're running inside poetry's virtualenv
def is_in_virtualenv():
    """Check if running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

# If not in virtualenv, re-execute with poetry
if not is_in_virtualenv():
    # Check if poetry is installed
    poetry_path = shutil.which('poetry')
    
    if not poetry_path:
        print("ERROR: Poetry is not installed.")
        print()
        print("Please install Poetry first:")
        print("  curl -sSL https://install.python-poetry.org | python3 -")
        print()
        print("Or visit: https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Re-execute this script with poetry run
    cmd = ['poetry', 'run', 'python'] + sys.argv
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: Failed to run with poetry: {e}")
        sys.exit(1)

# Now we're in the virtualenv, continue with normal imports
import argparse
from typing import Optional
from dotenv import load_dotenv, set_key, find_dotenv
from src.core.syslog2 import syslog2, setup_log, LogLevel

# Add project root to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir  # legale.py is in project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ProfileManager:
    """Manages bot profiles and their configurations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profiles_dir = project_root / "profiles"
        self.env_file = project_root / ".env"
        
    def get_current_profile(self) -> str:
        """Get the currently active profile name from .env file."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            profile = os.getenv("ACTIVE_PROFILE", "default")
        else:
            profile = "default"
        return profile
    
    def set_current_profile(self, profile_name: str):
        """Set the active profile in .env file."""
        env_path = str(self.env_file)
        
        # Create .env if it doesn't exist
        if not self.env_file.exists():
            self.env_file.touch()
        
        # Update or add ACTIVE_PROFILE
        set_key(env_path, "ACTIVE_PROFILE", profile_name)
        print(f"✓ Active profile set to: {profile_name}")
    
    def get_profile_dir(self, profile_name: Optional[str] = None) -> Path:
        """Get the directory path for a profile."""
        if profile_name is None:
            profile_name = self.get_current_profile()
        
        profile_dir = self.profiles_dir / profile_name
        return profile_dir
    
    def create_profile(self, profile_name: str, set_active: bool = False) -> Path:
        """Create a new profile directory structure."""
        profile_dir = self.get_profile_dir(profile_name)
        
        if profile_dir.exists():
            print(f"⚠️  Profile '{profile_name}' already exists at: {profile_dir}")
            return profile_dir
        
        # Create profile directory structure
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "chroma_db").mkdir(exist_ok=True)
        
        print(f"✓ Created profile '{profile_name}' at: {profile_dir}")
        print(f"  - Database will be: {profile_dir / 'legale_bot.db'}")
        print(f"  - Vector store will be: {profile_dir / 'chroma_db'}")
        
        if set_active:
            self.set_current_profile(profile_name)
        
        return profile_dir
    
    def list_profiles(self):
        """List all available profiles."""
        if not self.profiles_dir.exists():
            print("No profiles directory found. Create your first profile with:")
            print("  legale profile create <name>")
            return
        
        current = self.get_current_profile()
        profiles = [p.name for p in self.profiles_dir.iterdir() if p.is_dir()]
        
        if not profiles:
            print("No profiles found. Create your first profile with:")
            print("  legale profile create <name>")
            return
        
        print("Available profiles:")
        for profile in sorted(profiles):
            marker = " (active)" if profile == current else ""
            profile_dir = self.profiles_dir / profile
            db_path = profile_dir / "legale_bot.db"
            db_exists = "✓" if db_path.exists() else "✗"
            print(f"  {db_exists} {profile}{marker}")
        
        print(f"\nActive profile: {current}")
    
    def delete_profile(self, profile_name: str, force: bool = False):
        """Delete a profile and all its data."""
        if profile_name == self.get_current_profile() and not force:
            print(f"⚠️  Cannot delete active profile '{profile_name}'")
            print("   Switch to another profile first or use --force")
            return
        
        profile_dir = self.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            print(f"✗ Profile '{profile_name}' does not exist")
            return
        
        if not force:
            response = input(f"Delete profile '{profile_name}' and all its data? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled")
                return
        
        import shutil
        shutil.rmtree(profile_dir)
        print(f"✓ Deleted profile '{profile_name}'")
    
    def get_profile_paths(self, profile_name: Optional[str] = None) -> dict:
        """Get all relevant paths for a profile."""
        profile_dir = self.get_profile_dir(profile_name)
        
        return {
            'profile_dir': profile_dir,
            'db_path': profile_dir / 'legale_bot.db',
            'db_url': f"sqlite:///{profile_dir / 'legale_bot.db'}",
            'vector_db_path': profile_dir / 'chroma_db',
            'session_file': profile_dir / 'telegram_session.session',
        }


def cmd_profile(args, profile_manager: ProfileManager):
    """Handle profile management commands."""
    if args.profile_command == 'list':
        profile_manager.list_profiles()
    
    elif args.profile_command == 'create':
        profile_manager.create_profile(args.name, set_active=args.set_active)
    
    elif args.profile_command == 'switch':
        # Check if profile exists
        profile_dir = profile_manager.get_profile_dir(args.name)
        if not profile_dir.exists():
            print(f"✗ Profile '{args.name}' does not exist")
            print(f"  Create it with: legale profile create {args.name}")
            sys.exit(1)
        
        profile_manager.set_current_profile(args.name)
    
    elif args.profile_command == 'delete':
        profile_manager.delete_profile(args.name, force=args.force)
    
    elif args.profile_command == 'info':
        profile_name = args.name if args.name else profile_manager.get_current_profile()
        paths = profile_manager.get_profile_paths(profile_name)
        
        print(f"Profile: {profile_name}")
        print(f"  Directory: {paths['profile_dir']}")
        print(f"  Database: {paths['db_path']} ({'exists' if paths['db_path'].exists() else 'not created'})")
        print(f"  Vector DB: {paths['vector_db_path']} ({'exists' if paths['vector_db_path'].exists() else 'not created'})")
        print(f"  Session: {paths['session_file']} ({'exists' if paths['session_file'].exists() else 'not created'})")
    
    elif args.profile_command == 'option':
        cmd_profile_option(args, profile_manager)


def cmd_ingest(args, profile_manager: ProfileManager):
    """Handle data ingestion commands."""
    from src.ingestion.pipeline import IngestionPipeline
    
    # Get profile paths
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Ensure profile directory exists
    paths['profile_dir'].mkdir(parents=True, exist_ok=True)
    paths['vector_db_path'].mkdir(parents=True, exist_ok=True)
    
    print(f"Using profile: {profile_name}")
    print(f"Database: {paths['db_path']}")
    print(f"Vector store: {paths['vector_db_path']}")
    print()
    
    # Create pipeline with profile-specific paths
    pipeline = IngestionPipeline(
        db_url=paths['db_url'],
        vector_db_path=str(paths['vector_db_path']),
        profile_dir=str(paths['profile_dir'])
    )
    
    # Run ingestion
    pipeline.run(args.file, clear_existing=args.clear)


def cmd_telegram(args, profile_manager: ProfileManager):
    """Handle Telegram data fetching commands."""
    from src.ingestion.telegram import TelegramFetcher
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    if not API_ID or not API_HASH:
        print("✗ Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file")
        sys.exit(1)
    
    try:
        API_ID = int(API_ID)
    except ValueError:
        print("✗ Error: TELEGRAM_API_ID must be an integer")
        sys.exit(1)
    
    # Get profile-specific session file
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Ensure profile directory exists
    paths['profile_dir'].mkdir(parents=True, exist_ok=True)
    
    session_name = str(paths['session_file'].with_suffix(''))  # Remove .session extension
    
    print(f"Using profile: {profile_name}")
    print(f"Session file: {paths['session_file']}")
    print()
    
    fetcher = TelegramFetcher(API_ID, API_HASH, session_name=session_name)
    
    if args.telegram_command == 'list':
        fetcher.list_channels()
    
    elif args.telegram_command == 'members':
        fetcher.list_members(args.target)
    
    elif args.telegram_command == 'dump':
        # Default output to profile directory (will be set by dump_chat using chat ID)
        if not args.output:
            # Pass profile directory, dump_chat will form filename using chat ID
            args.output = str(paths['profile_dir'])
        else:
            # If output is specified, ensure it's a full path
            if not os.path.isabs(args.output):
                args.output = str(paths['profile_dir'] / args.output)
        
        fetcher.dump_chat(args.target, limit=args.limit, output_file=args.output)


def cmd_chat(args, profile_manager: ProfileManager):
    """Handle interactive chat commands."""
    from src.bot.cli import main as cli_main
    
    # Get profile paths
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Check if database exists
    if not paths['db_path'].exists():
        print(f"✗ Error: No database found for profile '{profile_name}'")
        print(f"  Create one by ingesting data first:")
        print(f"  legale ingest <file>")
        sys.exit(1)
    
    print(f"Using profile: {profile_name}")
    print(f"Database: {paths['db_path']}")
    print()
    
    # Set environment variables for the CLI
    os.environ['DATABASE_URL'] = paths['db_url']
    os.environ['VECTOR_DB_PATH'] = str(paths['vector_db_path'])
    os.environ['PROFILE_DIR'] = str(paths['profile_dir'])
    
    # Build CLI arguments
    cli_args = []
    
    # Handle Global -V if present (passed via args.log_level if we add it to main parser)
    if hasattr(args, 'log_level') and args.log_level:
        cli_args.extend(['-V', args.log_level])
        
    if args.verbose:
        cli_args.extend(['-' + 'v' * args.verbose])
    if args.chunks:
        cli_args.extend(['--chunks', str(args.chunks)])
    
    # Override sys.argv for the CLI
    original_argv = sys.argv
    sys.argv = ['cli.py'] + cli_args
    
    try:
        cli_main()
    finally:
        sys.argv = original_argv


def cmd_bot(args, profile_manager: ProfileManager):
    """Handle Telegram bot webhook commands."""
    from src.bot.tgbot import main as bot_main, register_webhook, delete_webhook, run_server, run_daemon
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get profile paths
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Set environment variables for the bot
    os.environ['DATABASE_URL'] = paths['db_url']
    os.environ['VECTOR_DB_PATH'] = str(paths['vector_db_path'])
    
    print(f"Using profile: {profile_name}")
    
    if args.bot_command == 'register':
        token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("✗ Error: TELEGRAM_BOT_TOKEN must be set in .env or passed via --token")
            sys.exit(1)
        
        register_webhook(args.url, token)
    
    elif args.bot_command == 'delete':
        token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("✗ Error: TELEGRAM_BOT_TOKEN must be set in .env or passed via --token")
            sys.exit(1)
        
        delete_webhook(token)
    
    elif args.bot_command == 'run':
        print(f"Database: {paths['db_path']}")
        print(f"Vector store: {paths['vector_db_path']}")
        print()
        
        # Respect global log_level if set
        if getattr(args, 'log_level', None):
            if args.log_level == 'DEBUG':
                args.verbose = max(args.verbose, 2)
            elif args.log_level == 'INFO':
                args.verbose = max(args.verbose, 1)
        
        run_server(args.host, args.port, args.verbose)
    
    elif args.bot_command == 'daemon':
        print(f"Database: {paths['db_path']}")
        print(f"Vector store: {paths['vector_db_path']}")
        print()
        
        run_daemon(args.host, args.port)


def cmd_test_embedding(args):
    """Test embedding generation for input text."""
    # NOTE: We intentionally allow online mode here for initial model download
    # After first download, the model will be cached and work offline
    
    from src.core.embedding import LocalEmbeddingClient
    import time
    
    print(f"Testing embedding generation with model: {args.model}")
    print(f"Text: '{args.text}'")
    
    try:
        # Initialize client and warmup model (exclude from timing)
        print("Loading model...")
        client = LocalEmbeddingClient(model=args.model)
        
        # Warmup: generate embedding once to load model into memory
        _ = client.get_embedding("warmup")
        print("Model loaded.\n")
        
        # Now measure only embedding generation time
        start_time = time.time()
        emb = client.get_embedding(args.text)
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"Success! ms={int(duration_ms)}")
        print(f"Dimensions: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    # ... existing code ...
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='legale',
        description='Legale Bot - Union Lawyer Chatbot with RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile management
  legale profile create mybot --set-active
  legale profile list
  legale profile switch mybot
  
  # Data ingestion
  legale telegram dump "My Chat" --limit 10000
  legale ingest telegram_dump_My_Chat.json
  
  # Interactive chat
  legale chat -vv
  
  # Telegram bot
  legale bot register --url https://example.com/webhook
  legale bot run -vv
  
For more information, visit: https://github.com/legale/rag-telegram-gpt-bot
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version='Legale Bot 1.0.0')
    parser.add_argument("-V", "--log-level", type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERR", "CRIT", "ALERT"],
        help="Set logging level (overrides -v)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ===== PROFILE MANAGEMENT =====
    profile_parser = subparsers.add_parser('profile', help='Manage bot profiles')
    profile_subparsers = profile_parser.add_subparsers(dest='profile_command', help='Profile command')
    
    # profile list
    profile_subparsers.add_parser('list', help='List all profiles')
    
    # profile create
    create_parser = profile_subparsers.add_parser('create', help='Create a new profile')
    create_parser.add_argument('name', help='Profile name')
    create_parser.add_argument('--set-active', action='store_true', help='Set as active profile')
    
    # profile switch
    switch_parser = profile_subparsers.add_parser('switch', help='Switch to a different profile')
    switch_parser.add_argument('name', help='Profile name')
    
    # profile delete
    delete_parser = profile_subparsers.add_parser('delete', help='Delete a profile')
    delete_parser.add_argument('name', help='Profile name')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    # profile info
    info_parser = profile_subparsers.add_parser('info', help='Show profile information')
    info_parser.add_argument('name', nargs='?', help='Profile name (default: current)')
    
    # profile option
    option_parser = profile_subparsers.add_parser('option', help='Manage profile options')
    option_parser.add_argument('option', choices=['model', 'generator', 'frequency'], help='Option to manage')
    option_parser.add_argument('action', choices=['list', 'get', 'set'], help='Action to perform')
    option_parser.add_argument('value', nargs='?', help='Value to set (required for set action)')
    option_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== DATA INGESTION =====
    ingest_parser = subparsers.add_parser('ingest', help='Ingest chat data into database')
    ingest_parser.add_argument('file', nargs='?', help='Path to chat dump file (JSON)')
    ingest_parser.add_argument('--clear', action='store_true', help='Clear existing data before ingestion')
    ingest_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== TELEGRAM FETCHING =====
    telegram_parser = subparsers.add_parser('telegram', help='Fetch data from Telegram')
    telegram_subparsers = telegram_parser.add_subparsers(dest='telegram_command', help='Telegram command')
    
    # telegram list
    list_parser = telegram_subparsers.add_parser('list', help='List all available chats')
    list_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # telegram members
    members_parser = telegram_subparsers.add_parser('members', help='List members of a chat')
    members_parser.add_argument('target', help='Chat ID or name')
    members_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # telegram dump
    dump_parser = telegram_subparsers.add_parser('dump', help='Dump chat messages')
    dump_parser.add_argument('target', help='Chat ID or name')
    dump_parser.add_argument('--limit', type=int, default=1000, help='Number of messages to fetch')
    dump_parser.add_argument('--output', help='Output file (default: profile_dir/telegram_dump_<target>.json)')
    dump_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== INTERACTIVE CHAT =====
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat session')
    chat_parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity level (-v, -vv, -vvv)')
    chat_parser.add_argument('--chunks', type=int, help='Number of context chunks to retrieve')
    chat_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== TELEGRAM BOT =====
    bot_parser = subparsers.add_parser('bot', help='Manage Telegram bot webhook')
    bot_subparsers = bot_parser.add_subparsers(dest='bot_command', help='Bot command')
    
    # bot register
    register_parser = bot_subparsers.add_parser('register', help='Register webhook with Telegram')
    register_parser.add_argument('--url', required=True, help='Webhook URL')
    register_parser.add_argument('--token', help='Bot token (or set TELEGRAM_BOT_TOKEN in .env)')
    register_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # bot delete
    delete_parser = bot_subparsers.add_parser('delete', help='Delete webhook from Telegram')
    delete_parser.add_argument('--token', help='Bot token (or set TELEGRAM_BOT_TOKEN in .env)')
    delete_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== CONFIGURATION =====
    config_parser = subparsers.add_parser('config', help='Manage profile configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config command')
    
    # config get
    get_config_parser = config_subparsers.add_parser('get', help='Get configuration value')
    get_config_parser.add_argument('key', choices=['system_prompt'], help='Configuration key')
    get_config_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # config set
    set_config_parser = config_subparsers.add_parser('set', help='Set configuration value')
    set_config_parser.add_argument('key', choices=['system_prompt'], help='Configuration key')
    set_config_parser.add_argument('value', help='Value to set')
    set_config_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # bot run
    run_parser = bot_subparsers.add_parser('run', help='Run bot in foreground mode')
    run_parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    run_parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    run_parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity level (-v, -vv, -vvv)')
    run_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # bot daemon
    # bot daemon
    daemon_parser = bot_subparsers.add_parser('daemon', help='Run bot as daemon')
    daemon_parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    daemon_parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    daemon_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # ===== TOPICS =====
    topics_parser = subparsers.add_parser('topics', help='Manage topics (clustering)')
    topics_subparsers = topics_parser.add_subparsers(dest='topic_command', help='Topic command')
    
    # topics build
    topics_build_parser = topics_subparsers.add_parser('build', help='Build topics from embeddings')
    topics_build_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # topics list
    topics_list_parser = topics_subparsers.add_parser('list', help='List existing topics')
    topics_list_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # topics show
    topics_show_parser = topics_subparsers.add_parser('show', help='Show details of a topic')
    topics_show_parser.add_argument('id', help='Topic ID')
    topics_show_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    
    # Test Embedding
    emb_parser = subparsers.add_parser('test-embedding', help='Test embedding generation')
    emb_parser.add_argument('text', help='Text to embed')
    emb_parser.add_argument('--model', default='ai-sage/Giga-Embeddings-instruct', help='Model name')

    # Parse arguments
    args = parser.parse_args()

    # Setup global logging
    syslog_level = LogLevel.LOG_WARNING
    if args.log_level:
        level_map = {
            "DEBUG": LogLevel.LOG_DEBUG,
            "INFO": LogLevel.LOG_INFO,
            "WARNING": LogLevel.LOG_WARNING,
            "ERR": LogLevel.LOG_ERR,
            "CRIT": LogLevel.LOG_CRIT,
            "ALERT": LogLevel.LOG_ALERT
        }
        syslog_level = level_map.get(args.log_level, LogLevel.LOG_WARNING)
    
    setup_log(syslog_level)
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Initialize profile manager
    profile_manager = ProfileManager(project_root)
    
    # Ensure default profile exists
    if args.command != 'profile':
        default_profile = profile_manager.get_profile_dir('default')
        if not default_profile.exists():
            print("Creating default profile...")
            profile_manager.create_profile('default', set_active=True)
            print()
    
    # Route to appropriate command handler
    if args.command == 'profile':
        if not args.profile_command:
            profile_parser.print_help()
            sys.exit(1)
        cmd_profile(args, profile_manager)
    
    elif args.command == 'ingest':
        if not args.file and not args.clear:
            ingest_parser.error("Please provide a file or specify --clear")
        cmd_ingest(args, profile_manager)
    
    elif args.command == 'telegram':
        if not args.telegram_command:
            telegram_parser.print_help()
            sys.exit(1)
        cmd_telegram(args, profile_manager)
    
    elif args.command == 'chat':
        cmd_chat(args, profile_manager)
    
    elif args.command == 'bot':
        if not args.bot_command:
            bot_parser.print_help()
            sys.exit(1)
        cmd_bot(args, profile_manager)

    elif args.command == 'config':
        if not args.config_command:
            config_parser.print_help()
            sys.exit(1)
        cmd_config(args, profile_manager)

    elif args.command == 'topics':
        if not args.topic_command:
            topics_parser.print_help()
            sys.exit(1)
        cmd_topics(args, profile_manager)

    elif args.command == 'test-embedding':
        cmd_test_embedding(args)


def cmd_topics(args, profile_manager: ProfileManager):
    """Handle topic management commands."""
    from src.ai.clustering import TopicClusterer
    from src.storage.db import Database
    from src.storage.vector_store import VectorStore
    from src.core.llm import LLMClient
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    if not paths['db_path'].exists():
        print(f"✗ Error: No database found for profile '{profile_name}'")
        sys.exit(1)
        
    db = Database(paths['db_url'])
    vector_store = VectorStore(persist_directory=str(paths['vector_db_path']))
    
    # Init LLM Client
    config = BotConfig(paths['profile_dir'])
    model_name = config.current_model
    if not model_name:
         # Fallback to models.txt
         try:
             if os.path.exists("models.txt"):
                 with open("models.txt", "r") as f:
                     model_name = f.readline().strip()
         except:
             pass
    if not model_name:
        model_name = "openai/gpt-3.5-turbo"
        
    llm_client = LLMClient(model=model_name, verbosity=args.verbose if hasattr(args, 'verbose') else 0)

    clusterer = TopicClusterer(db, vector_store, llm_client)

    if args.topic_command == 'build':
        print(f"Building topics for profile: {profile_name}")
        print("1. Running L1 Clustering (Fine-grained)...")
        clusterer.perform_l1_clustering()
        
        print("2. Running L2 Clustering (Super-topics)...")
        clusterer.perform_l2_clustering()
        
        print("3. Naming Topics (LLM)...")
        clusterer.name_topics()
        
        print("✓ Topic build complete.")
            
    elif args.topic_command == 'list':
        try:
            l2_topics = db.get_all_topics_l2()
            l1_topics = db.get_all_topics_l1()
            
            if not l1_topics:
                print("No topics found. Run 'legale topics build' first.")
                return
                
            print(f"\n{'ID':<5} {'L2 Topic Title':<40} {'L1 Count':<10} {'Chunks'}")
            print("-" * 75)
            
            l1_by_l2 = {}
            orphans = []
            for t in l1_topics:
                if t.parent_l2_id:
                    if t.parent_l2_id not in l1_by_l2:
                        l1_by_l2[t.parent_l2_id] = []
                    l1_by_l2[t.parent_l2_id].append(t)
                else:
                    orphans.append(t)
                    
            for l2 in l2_topics:
                children = l1_by_l2.get(l2.id, [])
                chunks_count = sum(c.chunk_count for c in children)
                title = l2.title or f"Topic L2-{l2.id}"
                print(f"{l2.id:<5} {title[:38]:<40} {len(children):<10} {chunks_count}")
                
            if orphans:
                print("\nOrphaned L1 Topics (No Super-Topic):")
                print(f"{'ID':<5} {'Title':<40} {'Chunks':<10}")
                print("-" * 60)
                for t in orphans:
                     print(f"{t.id:<5} {t.title[:38]:<40} {t.chunk_count:<10}")

                
        except Exception as e:
                print(f"✗ Error listing topics: {e}")
                
    elif args.topic_command == 'show':
        try:
            tid = int(args.id)
            l2 = next((t for t in db.get_all_topics_l2() if t.id == tid), None)
            
            if l2:
                print(f"=== Super-Topic L2-{l2.id} ===")
                print(f"Title: {l2.title}")
                print(f"Description: {l2.descr}")
                subtopics = db.get_l1_topics_by_l2(l2.id)
                print(f"Sub-topics count: {len(subtopics)}")
                
                print("\nSub-topics:")
                for sub in subtopics:
                    print(f"  [{sub.id}] {sub.title} ({sub.chunk_count} chunks)")
                return

            l1 = next((t for t in db.get_all_topics_l1() if t.id == tid), None)
            if l1:
                print(f"=== Topic L1-{l1.id} ===")
                print(f"Title: {l1.title}")
                print(f"Description: {l1.descr}")
                print(f"Parent L2: {l1.parent_l2_id}")
                print(f"Chunks: {l1.chunk_count}")
                print(f"Messages: {l1.msg_count}")
                print(f"Time: {l1.ts_from} - {l1.ts_to}")
                
                chunks = db.get_chunks_by_topic_l1(l1.id)
                print(f"\nSample Content ({min(3, len(chunks))} of {len(chunks)}):")
                for i, c in enumerate(chunks[:3]):
                    print(f"--- Chunk {i+1} ---")
                    print(c.text[:200].replace('\n', ' ') + "...")
                return
                
            print(f"Topic ID {tid} not found in L1 or L2 tables.")
                
        except ValueError:
            print("Error: Topic ID must be an integer.")
        except Exception as e:
            print(f"✗ Error showing topic: {e}")



def cmd_profile_option(args, profile_manager: ProfileManager):
    """Handle profile option management commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
        print(f"✗ Profile '{profile_name}' does not exist")
        sys.exit(1)
    
    config = BotConfig(profile_dir)
    option = args.option
    action = args.action
    
    # Define available values for each option
    available_models = {
        "openrouter": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ],
        "local": [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
    }
    
    all_models = available_models["openrouter"] + available_models["local"]
    available_generators = ["openrouter", "openai", "local"]
    
    if option == 'model':
        if action == 'list':
            print("Available embedding models:")
            print("\nOpenRouter/OpenAI models:")
            for model in available_models["openrouter"]:
                marker = " (current)" if model == config.embedding_model else ""
                print(f"  - {model}{marker}")
            print("\nLocal models (sentence-transformers):")
            for model in available_models["local"]:
                marker = " (current)" if model == config.embedding_model else ""
                print(f"  - {model}{marker}")
        
        elif action == 'get':
            print(f"Current embedding model: {config.embedding_model}")
        
        elif action == 'set':
            if not args.value:
                print("✗ Error: value is required for 'set' action")
                sys.exit(1)
            if args.value not in all_models:
                print(f"✗ Error: Unknown model '{args.value}'")
                print(f"  Available models: {', '.join(all_models)}")
                sys.exit(1)
            config.embedding_model = args.value
            print(f"✓ Embedding model set to: {args.value}")
    
    elif option == 'generator':
        if action == 'list':
            print("Available embedding generators:")
            for gen in available_generators:
                marker = " (current)" if gen == config.embedding_generator else ""
                print(f"  - {gen}{marker}")
            print("\nNote:")
            print("  - openrouter/openai: Use OpenRouter/OpenAI API")
            print("  - local: Use local sentence-transformers (no API key required)")
        
        elif action == 'get':
            print(f"Current embedding generator: {config.embedding_generator}")
        
        elif action == 'set':
            if not args.value:
                print("✗ Error: value is required for 'set' action")
                sys.exit(1)
            if args.value.lower() not in available_generators:
                print(f"✗ Error: Unknown generator '{args.value}'")
                print(f"  Available generators: {', '.join(available_generators)}")
                sys.exit(1)
            try:
                config.embedding_generator = args.value.lower()
                print(f"✓ Embedding generator set to: {args.value.lower()}")
            except ValueError as e:
                print(f"✗ Error: {e}")
                sys.exit(1)
    
    elif option == 'frequency':
        if action == 'list':
            print("Response frequency options:")
            print("  - 0: Respond only to mentions")
            print("  - 1: Respond to every message")
            print("  - N: Respond to every N-th message (N > 1)")
            print(f"\nCurrent value: {config.response_frequency}")
        
        elif action == 'get':
            freq = config.response_frequency
            if freq == 0:
                desc = "Respond only to mentions"
            elif freq == 1:
                desc = "Respond to every message"
            else:
                desc = f"Respond to every {freq}-th message"
            print(f"Current response frequency: {freq} ({desc})")
        
        elif action == 'set':
            if not args.value:
                print("✗ Error: value is required for 'set' action")
                sys.exit(1)
            try:
                freq_value = int(args.value)
                if freq_value < 0:
                    print("✗ Error: frequency must be >= 0")
                    sys.exit(1)
                config.response_frequency = freq_value
                print(f"✓ Response frequency set to: {freq_value}")
            except ValueError:
                print(f"✗ Error: frequency must be an integer")
                sys.exit(1)


def cmd_config(args, profile_manager: ProfileManager):
    """Handle configuration commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
         print(f"✗ Profile '{profile_name}' does not exist")
         sys.exit(1)
         
    config = BotConfig(profile_dir)
    
    if args.config_command == 'get':
        if args.key == 'system_prompt':
            prompt = config.system_prompt
            if not prompt:
                from src.core.prompt import PromptEngine
                prompt = f"(Default)\n{PromptEngine.SYSTEM_PROMPT_TEMPLATE}"
            print(f"System Prompt for profile '{profile_name}':\n\n{prompt}")
        else:
             print(f"Unknown config key: {args.key}")

    elif args.config_command == 'set':
        if args.key == 'system_prompt':
             config.system_prompt = args.value
             print(f"✓ System prompt updated for profile '{profile_name}'")
        else:
             print(f"Unknown config key: {args.key}")


if __name__ == '__main__':
    main()
