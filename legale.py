#!/usr/bin/env python3
"""
Legale Bot - Unified CLI Orchestrator

This is the main entry point for all Legale Bot operations.
It provides a unified interface for managing profiles, ingesting data,
running the bot, and managing the Telegram webhook.
"""

import sys
import os
import subprocess
import shutil
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
        vector_db_path=str(paths['vector_db_path'])
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
        # Default output to profile directory
        if not args.output:
            args.output = str(paths['profile_dir'] / f"telegram_dump_{args.target}.json")
        
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
    
    # Build CLI arguments
    cli_args = []
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
        
        run_server(args.host, args.port, args.verbose)
    
    elif args.bot_command == 'daemon':
        print(f"Database: {paths['db_path']}")
        print(f"Vector store: {paths['vector_db_path']}")
        print()
        
        run_daemon(args.host, args.port)


def main():
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
    
    # bot run
    run_parser = bot_subparsers.add_parser('run', help='Run bot in foreground mode')
    run_parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    run_parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    run_parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity level (-v, -vv, -vvv)')
    run_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # bot daemon
    daemon_parser = bot_subparsers.add_parser('daemon', help='Run bot as daemon')
    daemon_parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    daemon_parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    daemon_parser.add_argument('--profile', help='Profile to use (default: current active)')
    
    # Parse arguments
    args = parser.parse_args()
    
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


if __name__ == '__main__':
    main()
