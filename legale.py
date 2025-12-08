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
from src.core.syslog2 import *

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
import warnings
from typing import Optional
from dotenv import load_dotenv, set_key, find_dotenv
from src.core.syslog2 import syslog2, setup_log, LogLevel
from src.core.cli_parser import (
    CommandParser, CommandSpec, ArgStream, CLIError, CLIHelp,
    parse_flag, parse_option, parse_int_option, parse_float_option, parse_choice_option
)

# Suppress sklearn deprecation warnings from hdbscan and other libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Add project root to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir  # legale.py is in project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ProfileManager:
    """Manages bot profiles and their configurations."""
    
    # Default values for .env file
    ENV_DEFAULTS = {
        "ACTIVE_PROFILE": "default",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "MAX_CONTEXT_TOKENS": "14000"
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profiles_dir = project_root / "profiles"
        self.env_file = project_root / ".env"
        # Ensure .env has default values
        self._ensure_env_defaults()
    
    def _ensure_env_defaults(self):
        """Ensure .env file has all default values. Add missing ones."""
        env_path = str(self.env_file)
        
        # Create .env if it doesn't exist
        if not self.env_file.exists():
            self.env_file.touch()
        
        # Load current .env
        load_dotenv(self.env_file, override=False)
        
        # Add missing defaults
        updated = False
        for key, default_value in self.ENV_DEFAULTS.items():
            current_value = os.getenv(key)
            if current_value is None:
                set_key(env_path, key, default_value)
                updated = True
        
        # Reload if updated
        if updated:
            load_dotenv(self.env_file, override=True)
        
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
        print(f"Active profile set to: {profile_name}")
    
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
            print(f"Warning: Profile '{profile_name}' already exists at: {profile_dir}")
            return profile_dir
        
        # Create profile directory structure
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "chroma_db").mkdir(exist_ok=True)
        
        print(f"Created profile '{profile_name}' at: {profile_dir}")
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
            db_exists = "OK" if db_path.exists() else "MISSING"
            print(f"  {db_exists} {profile}{marker}")
        
        print(f"\nActive profile: {current}")
    
    def delete_profile(self, profile_name: str, force: bool = False):
        """Delete a profile and all its data."""
        if profile_name == self.get_current_profile() and not force:
            print(f"Warning: Cannot delete active profile '{profile_name}'")
            print("   set another profile first or user force flag")
            return
        
        profile_dir = self.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            print(f"Error: Profile '{profile_name}' does not exist")
            return
        
        if not force:
            response = input(f"Delete profile '{profile_name}' and all its data? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled")
                return
        
        import shutil
        shutil.rmtree(profile_dir)
        print(f"Deleted profile '{profile_name}'")
    
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
        profile_manager.create_profile(args.name, set_active=getattr(args, 'set_active', False))
    
    elif args.profile_command == 'get':
        # Return current profile name
        current_profile = profile_manager.get_current_profile()
        print(current_profile)
    
    elif args.profile_command == 'set':
        # Check if profile exists
        profile_dir = profile_manager.get_profile_dir(args.name)
        if not profile_dir.exists():
            print(f"Profile '{args.name}' does not exist")
            print(f"  Create it with: legale profile create {args.name}")
            sys.exit(1)
        
        profile_manager.set_current_profile(args.name)
    
    elif args.profile_command == 'delete':
        profile_manager.delete_profile(args.name, force=getattr(args, 'force', False))
    
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
    syslog2(LOG_WARNING, "ingest args:", **vars(args))
    
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
    
    # Route to appropriate subcommand
    ingest_command = getattr(args, 'ingest_command', None)
    
    # For 'info' command, we don't need the full pipeline - just DB and vector store count
    if ingest_command == 'info':
        from src.storage.db import Database
        
        db = Database(paths['db_url'])
        
        # Get database info
        db_info = db.get_database_info()
        
        # Get vector store count directly from ChromaDB without creating embedding client
        # Import chromadb only when needed to avoid initializing OpenAI SDK components
        try:
            import chromadb
            from chromadb.config import Settings
            
            chroma_client = chromadb.PersistentClient(
                path=str(paths['vector_db_path']),
                settings=Settings(anonymized_telemetry=False)
            )
            # Try to get collection, if it doesn't exist, count is 0
            try:
                collection = chroma_client.get_collection(name="default")
                vector_count = collection.count()
            except Exception:
                # Collection doesn't exist yet
                vector_count = 0
        except Exception:
            # Vector store directory doesn't exist or is inaccessible
            vector_count = 0
        
        # Format output: бд - таблица - кол-во записей
        db_name = paths['db_path'].name
        print(f"{'Database':30} {'Table':20} {'Records':>15}")
        print("-" * 65)
        
        # SQLite database tables
        for table_name, count in sorted(db_info.items()):
            print(f"{db_name:30} {table_name:20} {count:>15,}")
        
        # Vector store
        vec_db_name = paths['vector_db_path'].name if paths['vector_db_path'].is_dir() else str(paths['vector_db_path'])
        print(f"{vec_db_name:30} {'embeddings':20} {vector_count:>15,}")
        
        print()
        return
    
    # Create pipeline with profile-specific paths (for all other commands)
    pipeline = IngestionPipeline(
        db_url=paths['db_url'],
        vector_db_path=str(paths['vector_db_path']),
        profile_dir=str(paths['profile_dir'])
    )
    
    if ingest_command == 'all':
        if not args.file:
            print("Error: file path is required for 'ingest all'")
            sys.exit(1)
        
        # Normalize file path - convert relative to absolute
        file_path = args.file
        if not os.path.isabs(file_path):
            # If relative, resolve relative to current working directory
            file_path = os.path.abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        model = getattr(args, 'model', None)
        batch_size = getattr(args, 'batch_size', 128)
        clustering_params = {}
        if hasattr(args, 'min_cluster_size') and args.min_cluster_size:
            clustering_params['min_cluster_size'] = args.min_cluster_size
        if hasattr(args, 'min_samples') and args.min_samples:
            clustering_params['min_samples'] = args.min_samples
        if hasattr(args, 'metric') and args.metric:
            clustering_params['metric'] = args.metric
        if hasattr(args, 'cluster_selection_method') and args.cluster_selection_method:
            clustering_params['cluster_selection_method'] = args.cluster_selection_method
        if hasattr(args, 'cluster_selection_epsilon') and args.cluster_selection_epsilon is not None:
            clustering_params['cluster_selection_epsilon'] = args.cluster_selection_epsilon
        pipeline.run_all(file_path, model=model, batch_size=batch_size, **clustering_params)
        
    elif ingest_command == 'stage0':
        if not args.file:
            print("Error: file path is required for 'ingest stage0'")
            sys.exit(1)
        
        # Normalize file path - convert relative to absolute
        file_path = args.file
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        print("Running stage0: Parse and store...")
        pipeline.run_stage0(file_path)
        print("Stage0 complete.")
        
    elif ingest_command == 'stage1':
        # Check if there are chunks in database
        from src.storage.db import Database
        db = Database(paths['db_url'])
        chunk_count = db.count_chunks()
        if chunk_count == 0:
            print("Error: No chunks found in database. Run 'ingest stage0 <file>' first.")
            sys.exit(1)
        
        model = getattr(args, 'model', None)
        batch_size = getattr(args, 'batch_size', 128)
        print("Running stage1: Generate embeddings...")
        pipeline.run_stage1(model=model, batch_size=batch_size)
        print("Stage1 complete.")
        
    elif ingest_command == 'stage2':
        # Check if there are embeddings
        vector_count = pipeline.vector_store.count()
        if vector_count == 0:
            print("Error: No embeddings found. Run 'ingest stage1' first.")
            sys.exit(1)
        
        clustering_params = {}
        if hasattr(args, 'min_cluster_size') and args.min_cluster_size:
            clustering_params['min_cluster_size'] = args.min_cluster_size
        if hasattr(args, 'min_samples') and args.min_samples:
            clustering_params['min_samples'] = args.min_samples
        if hasattr(args, 'metric') and args.metric:
            clustering_params['metric'] = args.metric
        if hasattr(args, 'cluster_selection_method') and args.cluster_selection_method:
            clustering_params['cluster_selection_method'] = args.cluster_selection_method
        if hasattr(args, 'cluster_selection_epsilon') and args.cluster_selection_epsilon is not None:
            clustering_params['cluster_selection_epsilon'] = args.cluster_selection_epsilon
        
        print("Running stage2: Cluster L1 topics...")
        pipeline.run_stage2(**clustering_params)
        print("Stage2 complete.")
        
    elif ingest_command == 'stage3':
        # Check if there are L1 topics
        from src.storage.db import Database
        db = Database(paths['db_url'])
        l1_topics = db.get_all_topics_l1()
        if not l1_topics:
            print("Error: No L1 topics found. Run 'ingest stage2' first.")
            sys.exit(1)
        
        print("Running stage3: Create embeddings for clusters and assign topics...")
        pipeline.run_stage3()
        print("Stage3 complete.")
        
    elif ingest_command == 'stage4':
        # Check if there are L1 topics
        from src.storage.db import Database, ChunkModel
        db = Database(paths['db_url'])
        l1_topics = db.get_all_topics_l1()
        if not l1_topics:
            print("Error: No L1 topics found. Run 'ingest stage2' first.")
            sys.exit(1)
        
        # Check if topics are assigned to chunks
        session = db.get_session()
        try:
            chunks_with_topics = session.query(ChunkModel).filter(ChunkModel.topic_l1_id.isnot(None)).count()
            if chunks_with_topics == 0:
                print("Error: No chunks have topic assignments. Run 'ingest stage3' first.")
                sys.exit(1)
        finally:
            session.close()
        
        only_unnamed = getattr(args, 'only_unnamed', False)
        rebuild = getattr(args, 'rebuild', False)
        print("Running stage4: Name L1 topics...")
        pipeline.run_stage4(only_unnamed=only_unnamed, rebuild=rebuild)
        print("Stage4 complete.")
        
    elif ingest_command == 'stage5':
        # Check if there are L1 topics
        from src.storage.db import Database
        db = Database(paths['db_url'])
        l1_topics = db.get_all_topics_l1()
        if not l1_topics:
            print("Error: No L1 topics found. Run 'ingest stage2' first.")
            sys.exit(1)
        
        clustering_params = {}
        if hasattr(args, 'min_cluster_size') and args.min_cluster_size:
            clustering_params['min_cluster_size'] = args.min_cluster_size
        if hasattr(args, 'min_samples') and args.min_samples:
            clustering_params['min_samples'] = args.min_samples
        if hasattr(args, 'metric') and args.metric:
            clustering_params['metric'] = args.metric
        if hasattr(args, 'cluster_selection_method') and args.cluster_selection_method:
            clustering_params['cluster_selection_method'] = args.cluster_selection_method
        if hasattr(args, 'cluster_selection_epsilon') and args.cluster_selection_epsilon is not None:
            clustering_params['cluster_selection_epsilon'] = args.cluster_selection_epsilon
        
        print("Running stage5: Cluster L2 topics and name...")
        pipeline.run_stage5(**clustering_params)
        print("Stage5 complete.")
        
    elif ingest_command == 'clear_all':
        print("Clearing all stages...")
        pipeline.clear_all()
        print("All stages cleared.")
        
    elif ingest_command == 'clear_stage0':
        print("Clearing stage0: messages...")
        deleted = pipeline.clear_stage0()
        print(f"Stage0 cleared: {deleted} messages deleted.")
        
    elif ingest_command == 'clear_stage1':
        print("Clearing stage1: embeddings...")
        removed = pipeline.clear_stage1()
        print(f"Stage1 cleared: {removed} embeddings removed.")
        
    elif ingest_command == 'clear_stage2':
        print("Clearing stage2: topics_l1...")
        deleted = pipeline.clear_stage2()
        print(f"Stage2 cleared: {deleted} topics_l1 deleted.")
        
    elif ingest_command == 'clear_stage3':
        print("Clearing stage3: topic_l1_id assignments...")
        updated = pipeline.clear_stage3()
        print(f"Stage3 cleared: {updated} chunks updated.")
        
    elif ingest_command == 'clear_stage4':
        print("Clearing stage4: topic names (no data to clear)...")
        pipeline.clear_stage4()
        print("Stage4 cleared.")
        
    elif ingest_command == 'clear_stage5':
        print("Clearing stage5: topic_l2_id assignments and topics_l2...")
        result = pipeline.clear_stage5()
        print(f"Stage5 cleared: {result} items updated/deleted.")
    
    else:
        # Should not happen due to routing logic, but handle gracefully
        print("Error: Unknown ingest command")
        sys.exit(1)


def cmd_telegram(args, profile_manager: ProfileManager):
    """Handle Telegram data fetching commands."""
    from src.ingestion.telegram import TelegramFetcher
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file")
        sys.exit(1)
    
    try:
        API_ID = int(API_ID)
    except ValueError:
        print("Error: TELEGRAM_API_ID must be an integer")
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
    
    elif args.telegram_command == 'ingest_all':
        # Find chat first to get its ID for filename
        # We need to connect to get chat info, but dump_chat will connect again
        # So we'll just let dump_chat handle the connection and find the file afterwards
        output_dir = str(paths['profile_dir'])
        
        # Dump chat (this will create telegram_dump_{chat_id}.json in output_dir)
        fetcher.dump_chat(args.target, limit=args.limit, output_file=output_dir)
        
        # Find the dumped file (dump_chat creates telegram_dump_{chat_id}.json)
        import glob
        dump_files = glob.glob(os.path.join(output_dir, "telegram_dump_*.json"))
        if not dump_files:
            print("Error: Failed to find dumped file")
            sys.exit(1)
        
        # Use the most recent dump file
        dump_file = max(dump_files, key=os.path.getmtime)
        
        # Check if dump file was created
        if not os.path.exists(dump_file):
            print(f"Error: Failed to create dump file: {dump_file}")
            sys.exit(1)
        
        print(f"\nStarting ingestion of {dump_file}...\n")
        
        # Now run ingest all on the dumped file
        from src.ingestion.pipeline import IngestionPipeline
        
        # Ensure profile directory exists
        paths['profile_dir'].mkdir(parents=True, exist_ok=True)
        paths['vector_db_path'].mkdir(parents=True, exist_ok=True)
        
        # Create pipeline with profile-specific paths
        pipeline = IngestionPipeline(
            db_url=paths['db_url'],
            vector_db_path=str(paths['vector_db_path']),
            profile_dir=str(paths['profile_dir'])
        )
        
        # Run ingest all
        model = getattr(args, 'model', None)
        batch_size = getattr(args, 'batch_size', 128)
        pipeline.run_all(dump_file, model=model, batch_size=batch_size)


def cmd_chat(args, profile_manager: ProfileManager):
    """Handle interactive chat commands."""
    from src.bot.cli import main as cli_main
    
    # Get profile paths
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Check if database exists
    if not paths['db_path'].exists():
        print(f"Error: No database found for profile '{profile_name}'")
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
    
    # Handle Global -V if present (passed via args.log_level)
    if hasattr(args, 'log_level') and args.log_level:
        cli_args.extend(['-V', args.log_level])
        
    if hasattr(args, 'chunks') and args.chunks:
        cli_args.extend(['--chunks', str(args.chunks)])
    if hasattr(args, 'debug_rag') and args.debug_rag:
        cli_args.append('--debug-rag')
    
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
            print("Error: TELEGRAM_BOT_TOKEN must be set in .env or passed via --token")
            sys.exit(1)
        
        register_webhook(args.url, token)
    
    elif args.bot_command == 'delete':
        token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("Error: TELEGRAM_BOT_TOKEN must be set in .env or passed via --token")
            sys.exit(1)
        
        delete_webhook(token)
    
    elif args.bot_command == 'run':
        print(f"Database: {paths['db_path']}")
        print(f"Vector store: {paths['vector_db_path']}")
        print()
        
        # Map log_level to verbosity for run_server
        verbose = 0
        log_level = getattr(args, 'log_level', None)
        if log_level:
            if log_level in ('LOG_DEBUG', 'LOG_INFO'):
                verbose = 2
            elif log_level == 'LOG_WARNING':
                verbose = 1
        
        run_server(args.host, args.port, verbose)
    
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
    
    model = getattr(args, 'model', 'ai-sage/Giga-Embeddings-instruct')
    text = getattr(args, 'text', '')
    
    print(f"Testing embedding generation with model: {model}")
    print(f"Text: '{text}'")
    
    try:
        # Initialize client and warmup model (exclude from timing)
        print("Loading model...")
        client = LocalEmbeddingClient(model=model)
        
        # Warmup: generate embedding once to load model into memory
        _ = client.get_embedding("warmup")
        print("Model loaded.\n")
        
        # Now measure only embedding generation time
        start_time = time.time()
        emb = client.get_embedding(text)
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"Success! ms={int(duration_ms)}")
        print(f"Dimensions: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
    except Exception as e:
        from src.core.syslog2 import syslog2, LogLevel, setup_log
        setup_log(LogLevel.LOG_WARNING)
        syslog2(LogLevel.LOG_ERR, f"Error: {e}")
        sys.exit(1)


# ===== PARSER FUNCTIONS FOR COMMANDPARSER =====

def parse_test_embedding(stream: ArgStream) -> dict:
    """Parse test-embedding command."""
    text = stream.expect("text")
    model = parse_option(stream, "model") or "ai-sage/Giga-Embeddings-instruct"
    return {"text": text, "model": model}


def parse_profile_list(stream: ArgStream) -> dict:
    """Parse profile list command."""
    return {"profile": parse_option(stream, "profile")}


def parse_profile_create(stream: ArgStream) -> dict:
    """Parse profile create command."""
    name = stream.expect("profile name")
    set_active = parse_flag(stream, "set-active")
    return {"name": name, "set_active": set_active, "profile": parse_option(stream, "profile")}


def parse_profile_get(stream: ArgStream) -> dict:
    """Parse profile get command - returns current profile name."""
    # No arguments needed, just return empty dict
    return {"profile": parse_option(stream, "profile")}


def parse_profile_set(stream: ArgStream) -> dict:
    """Parse profile set command."""
    name = stream.expect("profile name")
    return {"name": name, "profile": parse_option(stream, "profile")}


def parse_profile_delete(stream: ArgStream) -> dict:
    """Parse profile delete command."""
    name = stream.expect("profile name")
    force = parse_flag(stream, "force")
    return {"name": name, "force": force, "profile": parse_option(stream, "profile")}


def parse_profile_info(stream: ArgStream) -> dict:
    """Parse profile info command."""
    name = None
    if stream.has_next() and stream.peek() != "profile":
        name = stream.next()
    profile = parse_option(stream, "profile")
    return {"name": name, "profile": profile}


def parse_profile_option(stream: ArgStream) -> dict:
    """Parse profile option command."""
    option = stream.expect("option")
    action = stream.expect("action")
    value = None
    if stream.has_next() and stream.peek() not in ("profile",):
        value = stream.next()
    profile = parse_option(stream, "profile")
    return {"option": option, "action": action, "value": value, "profile": profile}


def parse_ingest_all(stream: ArgStream) -> dict:
    """Parse ingest all command."""
    file = stream.expect("file path")
    model = parse_option(stream, "model")
    batch_size = parse_int_option(stream, "batch-size", 128)
    profile = parse_option(stream, "profile")
    return {"file": file, "model": model, "batch_size": batch_size, "profile": profile, "ingest_command": "all"}


def parse_ingest_stage0(stream: ArgStream) -> dict:
    """Parse ingest stage0 command."""
    file = stream.expect("file path")
    profile = parse_option(stream, "profile")
    return {"file": file, "profile": profile, "ingest_command": "stage0"}


def parse_ingest_stage1(stream: ArgStream) -> dict:
    """Parse ingest stage1 command."""
    model = parse_option(stream, "model")
    batch_size = parse_int_option(stream, "batch-size", 128)
    profile = parse_option(stream, "profile")
    return {"model": model, "batch_size": batch_size, "profile": profile, "ingest_command": "stage1"}


def parse_ingest_stage2(stream: ArgStream) -> dict:
    """Parse ingest stage2 command."""
    min_cluster_size = parse_int_option(stream, "min-cluster-size", 2)
    min_samples = parse_int_option(stream, "min-samples", 1)
    metric = parse_option(stream, "metric")
    cluster_selection_method = parse_option(stream, "cluster-selection-method")
    cluster_selection_epsilon = parse_float_option(stream, "cluster-selection-epsilon", 0.0)
    profile = parse_option(stream, "profile")
    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "profile": profile,
        "ingest_command": "stage2"
    }


def parse_ingest_stage3(stream: ArgStream) -> dict:
    """Parse ingest stage3 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "stage3"}


def parse_ingest_stage4(stream: ArgStream) -> dict:
    """Parse ingest stage4 command."""
    only_unnamed = parse_flag(stream, "only-unnamed")
    rebuild = parse_flag(stream, "rebuild")
    profile = parse_option(stream, "profile")
    return {"only_unnamed": only_unnamed, "rebuild": rebuild, "profile": profile, "ingest_command": "stage4"}


def parse_ingest_stage5(stream: ArgStream) -> dict:
    """Parse ingest stage5 command."""
    min_cluster_size = parse_int_option(stream, "min-cluster-size", 2)
    min_samples = parse_int_option(stream, "min-samples", 1)
    metric = parse_option(stream, "metric")
    cluster_selection_method = parse_option(stream, "cluster-selection-method")
    cluster_selection_epsilon = parse_float_option(stream, "cluster-selection-epsilon", 0.0)
    profile = parse_option(stream, "profile")
    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "profile": profile,
        "ingest_command": "stage5"
    }


def parse_ingest_clear_all(stream: ArgStream) -> dict:
    """Parse ingest clear all command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_all"}


def parse_ingest_clear_stage0(stream: ArgStream) -> dict:
    """Parse ingest clear stage0 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage0"}


def parse_ingest_clear_stage1(stream: ArgStream) -> dict:
    """Parse ingest clear stage1 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage1"}


def parse_ingest_clear_stage2(stream: ArgStream) -> dict:
    """Parse ingest clear stage2 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage2"}


def parse_ingest_clear_stage3(stream: ArgStream) -> dict:
    """Parse ingest clear stage3 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage3"}


def parse_ingest_clear_stage4(stream: ArgStream) -> dict:
    """Parse ingest clear stage4 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage4"}


def parse_ingest_clear_stage5(stream: ArgStream) -> dict:
    """Parse ingest clear stage5 command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "clear_stage5"}


def parse_ingest_info(stream: ArgStream) -> dict:
    """Parse ingest info command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "ingest_command": "info"}


def parse_ingest(stream: ArgStream) -> dict:
    """Parse ingest command without subcommand (treats as 'all')."""
    if not stream.has_next():
        raise CLIError("ingest subcommand required (all, stage0-4, clear all/stage0-4, info) or file path")
    
    # Check if first argument is a subcommand
    first = stream.peek().lower()
    if first in ("all", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "clear", "info"):
        raise CLIError(f"ingest subcommand '{first}' requires explicit subcommand syntax")
    
    # Treat as 'all' with file path
    file = stream.next()
    model = parse_option(stream, "model")
    batch_size = parse_int_option(stream, "batch-size", 128)
    profile = parse_option(stream, "profile")
    return {"file": file, "model": model, "batch_size": batch_size, "profile": profile, "ingest_command": "all"}




def parse_telegram_list(stream: ArgStream) -> dict:
    """Parse telegram list command."""
    return {"profile": parse_option(stream, "profile"), "telegram_command": "list"}


def parse_telegram_members(stream: ArgStream) -> dict:
    """Parse telegram members command."""
    target = stream.expect("target")
    profile = parse_option(stream, "profile")
    return {"target": target, "profile": profile, "telegram_command": "members"}


def parse_telegram_dump(stream: ArgStream) -> dict:
    """Parse telegram dump command."""
    target = stream.expect("target")
    limit = parse_int_option(stream, "limit", 1000)
    output = parse_option(stream, "output")
    profile = parse_option(stream, "profile")
    return {"target": target, "limit": limit, "output": output, "profile": profile, "telegram_command": "dump"}


def parse_telegram_ingest_all(stream: ArgStream) -> dict:
    """Parse telegram ingest all command."""
    target = stream.expect("target")
    limit = parse_int_option(stream, "limit", 1000)
    model = parse_option(stream, "model")
    batch_size = parse_int_option(stream, "batch-size", 128)
    profile = parse_option(stream, "profile")
    return {"target": target, "limit": limit, "model": model, "batch_size": batch_size, "profile": profile, "telegram_command": "ingest_all"}


def parse_chat(stream: ArgStream) -> dict:
    """Parse chat command."""
    chunks = parse_int_option(stream, "chunks")
    debug_rag = parse_flag(stream, "debug-rag")
    profile = parse_option(stream, "profile")
    return {"chunks": chunks, "debug_rag": debug_rag, "profile": profile}


def parse_bot_register(stream: ArgStream) -> dict:
    """Parse bot register command."""
    url = parse_option(stream, "url")
    if not url:
        raise CLIError("url required for bot register")
    token = parse_option(stream, "token")
    profile = parse_option(stream, "profile")
    return {"url": url, "token": token, "profile": profile, "bot_command": "register"}


def parse_bot_delete(stream: ArgStream) -> dict:
    """Parse bot delete command."""
    token = parse_option(stream, "token")
    profile = parse_option(stream, "profile")
    return {"token": token, "profile": profile, "bot_command": "delete"}


def parse_bot_run(stream: ArgStream) -> dict:
    """Parse bot run command."""
    host = parse_option(stream, "host") or "127.0.0.1"
    port = parse_int_option(stream, "port", 8000)
    profile = parse_option(stream, "profile")
    return {"host": host, "port": port, "profile": profile, "bot_command": "run"}


def parse_bot_daemon(stream: ArgStream) -> dict:
    """Parse bot daemon command."""
    host = parse_option(stream, "host") or "127.0.0.1"
    port = parse_int_option(stream, "port", 8000)
    profile = parse_option(stream, "profile")
    return {"host": host, "port": port, "profile": profile, "bot_command": "daemon"}


def parse_config_get(stream: ArgStream) -> dict:
    """Parse config get command."""
    key = stream.expect("key")
    profile = parse_option(stream, "profile")
    return {"key": key, "profile": profile, "config_command": "get"}


def parse_config_set(stream: ArgStream) -> dict:
    """Parse config set command."""
    key = stream.expect("key")
    value = stream.expect("value")
    profile = parse_option(stream, "profile")
    return {"key": key, "value": value, "profile": profile, "config_command": "set"}


def parse_clustering_params(stream: ArgStream, prefix: str) -> dict:
    """Parse clustering parameters for a given prefix."""
    return {
        f"{prefix}_min_size": parse_int_option(stream, f"{prefix}-min-size", 2),
        f"{prefix}_min_samples": parse_int_option(stream, f"{prefix}-min-samples", 1),
        f"{prefix}_metric": parse_choice_option(stream, f"{prefix}-metric", ["cosine", "euclidean", "manhattan"], "cosine"),
        f"{prefix}_method": parse_choice_option(stream, f"{prefix}-method", ["eom", "leaf"], "eom"),
        f"{prefix}_epsilon": parse_float_option(stream, f"{prefix}-epsilon", 0.0),
    }


def parse_topics_cluster_l1(stream: ArgStream) -> dict:
    """Parse topics cluster-l1 command."""
    params = parse_clustering_params(stream, "l1")
    profile = parse_option(stream, "profile")
    params.update({"profile": profile, "topic_command": "cluster-l1"})
    return params


def parse_topics_cluster_l2(stream: ArgStream) -> dict:
    """Parse topics cluster-l2 command."""
    params = parse_clustering_params(stream, "l2")
    profile = parse_option(stream, "profile")
    params.update({"profile": profile, "topic_command": "cluster-l2"})
    return params


def parse_topics_name(stream: ArgStream) -> dict:
    """Parse topics name command."""
    only_unnamed = parse_flag(stream, "only-unnamed")
    rebuild = parse_flag(stream, "rebuild")
    target = parse_choice_option(stream, "target", ["l1", "l2", "both"], "both")
    profile = parse_option(stream, "profile")
    return {"only_unnamed": only_unnamed, "rebuild": rebuild, "target": target, "profile": profile, "topic_command": "name"}


def parse_topics_build(stream: ArgStream) -> dict:
    """Parse topics build command."""
    params = {}
    params.update(parse_clustering_params(stream, "l1"))
    params.update(parse_clustering_params(stream, "l2"))
    profile = parse_option(stream, "profile")
    params.update({"profile": profile, "topic_command": "build"})
    return params


def parse_topics_list(stream: ArgStream) -> dict:
    """Parse topics list command."""
    profile = parse_option(stream, "profile")
    return {"profile": profile, "topic_command": "list"}


def parse_topics_show(stream: ArgStream) -> dict:
    """Parse topics show command."""
    id_str = stream.expect("topic id")
    try:
        topic_id = int(id_str)
    except ValueError:
        raise CLIError(f"invalid topic id: {id_str}")
    profile = parse_option(stream, "profile")
    return {"id": str(topic_id), "profile": profile, "topic_command": "show"}


# Subcommand parsers
def _parse_profile_subcommand(stream: ArgStream) -> dict:
    """Parse profile subcommand."""
    if not stream.has_next():
        raise CLIError("profile subcommand required (list, create, get, set, delete, info, option)")
    subcmd = stream.next().lower()
    
    result = None
    if subcmd == "list":
        result = parse_profile_list(stream)
        result["profile_command"] = "list"
    elif subcmd == "create":
        result = parse_profile_create(stream)
        result["profile_command"] = "create"
    elif subcmd == "get":
        result = parse_profile_get(stream)
        result["profile_command"] = "get"
    elif subcmd == "set":
        result = parse_profile_set(stream)
        result["profile_command"] = "set"
    elif subcmd == "delete":
        result = parse_profile_delete(stream)
        result["profile_command"] = "delete"
    elif subcmd == "info":
        result = parse_profile_info(stream)
        result["profile_command"] = "info"
    elif subcmd == "option":
        result = parse_profile_option(stream)
        result["profile_command"] = "option"
    else:
        raise CLIError(f"unknown profile subcommand: {subcmd}")
    
    return result


def _parse_profile_subcommand_with_args(args):
    """Extract profile subcommand from args."""
    profile_command = getattr(args, 'profile_command', None)
    if not profile_command:
        raise CLIError("profile subcommand required")
    return f"profile_{profile_command}", args


def _parse_ingest_subcommand(stream: ArgStream) -> dict:
    """Parse ingest subcommand."""
    if not stream.has_next():
        # Show help if no arguments
        raise CLIHelp()
    
    subcmd = stream.next().lower()
    if subcmd == "all":
        return parse_ingest_all(stream)
    elif subcmd == "stage0":
        return parse_ingest_stage0(stream)
    elif subcmd == "stage1":
        return parse_ingest_stage1(stream)
    elif subcmd == "stage2":
        return parse_ingest_stage2(stream)
    elif subcmd == "stage3":
        return parse_ingest_stage3(stream)
    elif subcmd == "stage4":
        return parse_ingest_stage4(stream)
    elif subcmd == "stage5":
        return parse_ingest_stage5(stream)
    elif subcmd == "clear":
        # Parse clear subcommand
        if not stream.has_next():
            raise CLIError("clear subcommand required (all, stage0-4)")
        clear_subcmd = stream.next().lower()
        if clear_subcmd == "all":
            return parse_ingest_clear_all(stream)
        elif clear_subcmd == "stage0":
            return parse_ingest_clear_stage0(stream)
        elif clear_subcmd == "stage1":
            return parse_ingest_clear_stage1(stream)
        elif clear_subcmd == "stage2":
            return parse_ingest_clear_stage2(stream)
        elif clear_subcmd == "stage3":
            return parse_ingest_clear_stage3(stream)
        elif clear_subcmd == "stage4":
            return parse_ingest_clear_stage4(stream)
        elif clear_subcmd == "stage5":
            return parse_ingest_clear_stage5(stream)
        else:
            raise CLIError(f"unknown clear subcommand: {clear_subcmd}. Use: all, stage0, stage1, stage2, stage3, stage4, stage5")
    elif subcmd == "info":
        return parse_ingest_info(stream)
    else:
        raise CLIError(f"unknown ingest subcommand: {subcmd}. Use: all, stage0, stage1, stage2, stage3, stage4, stage5, clear all/stage0/stage1/stage2/stage3/stage4/stage5, info")


def _parse_ingest_subcommand_with_args(args):
    """Extract ingest subcommand from args."""
    ingest_command = getattr(args, 'ingest_command', 'all')
    return f"ingest_{ingest_command}", args


def _parse_telegram_subcommand(stream: ArgStream) -> dict:
    """Parse telegram subcommand."""
    if not stream.has_next():
        raise CLIError("telegram subcommand required (list, members, dump, ingest)")
    subcmd = stream.next().lower()
    
    result = None
    if subcmd == "list":
        result = parse_telegram_list(stream)
    elif subcmd == "members":
        result = parse_telegram_members(stream)
    elif subcmd == "dump":
        result = parse_telegram_dump(stream)
    elif subcmd == "ingest":
        # Parse ingest subcommand
        if not stream.has_next():
            raise CLIError("telegram ingest subcommand required (all)")
        ingest_subcmd = stream.next().lower()
        if ingest_subcmd == "all":
            result = parse_telegram_ingest_all(stream)
        else:
            raise CLIError(f"unknown telegram ingest subcommand: {ingest_subcmd}")
    else:
        raise CLIError(f"unknown telegram subcommand: {subcmd}")
    
    # telegram_command already set by parsers
    return result


def _parse_telegram_subcommand_with_args(args):
    """Extract telegram subcommand from args."""
    telegram_command = getattr(args, 'telegram_command', None)
    if not telegram_command:
        raise CLIError("telegram subcommand required")
    return f"telegram_{telegram_command}", args


def _parse_bot_subcommand(stream: ArgStream) -> dict:
    """Parse bot subcommand."""
    if not stream.has_next():
        raise CLIError("bot subcommand required (register, delete, run, daemon)")
    subcmd = stream.next().lower()
    
    result = None
    if subcmd == "register":
        result = parse_bot_register(stream)
    elif subcmd == "delete":
        result = parse_bot_delete(stream)
    elif subcmd == "run":
        result = parse_bot_run(stream)
    elif subcmd == "daemon":
        result = parse_bot_daemon(stream)
    else:
        raise CLIError(f"unknown bot subcommand: {subcmd}")
    
    # bot_command already set by parsers
    return result


def _parse_bot_subcommand_with_args(args):
    """Extract bot subcommand from args."""
    bot_command = getattr(args, 'bot_command', None)
    if not bot_command:
        raise CLIError("bot subcommand required")
    return f"bot_{bot_command}", args


def _parse_config_subcommand(stream: ArgStream) -> dict:
    """Parse config subcommand."""
    if not stream.has_next():
        raise CLIError("config subcommand required (get, set)")
    subcmd = stream.next().lower()
    
    result = None
    if subcmd == "get":
        result = parse_config_get(stream)
    elif subcmd == "set":
        result = parse_config_set(stream)
    else:
        raise CLIError(f"unknown config subcommand: {subcmd}")
    
    # config_command already set by parsers
    return result


def _parse_config_subcommand_with_args(args):
    """Extract config subcommand from args."""
    config_command = getattr(args, 'config_command', None)
    if not config_command:
        raise CLIError("config subcommand required")
    return f"config_{config_command}", args


def _parse_topics_subcommand(stream: ArgStream) -> dict:
    """Parse topics subcommand."""
    if not stream.has_next():
        raise CLIError("topics subcommand required (cluster-l1, cluster-l2, name, build, list, show)")
    subcmd = stream.next().lower()
    
    result = None
    if subcmd == "cluster-l1":
        result = parse_topics_cluster_l1(stream)
    elif subcmd == "cluster-l2":
        result = parse_topics_cluster_l2(stream)
    elif subcmd == "name":
        result = parse_topics_name(stream)
    elif subcmd == "build":
        result = parse_topics_build(stream)
    elif subcmd == "list":
        result = parse_topics_list(stream)
    elif subcmd == "show":
        result = parse_topics_show(stream)
    else:
        raise CLIError(f"unknown topics subcommand: {subcmd}")
    
    # topic_command already set by parsers
    return result


def _parse_topics_subcommand_with_args(args):
    """Extract topics subcommand from args."""
    topic_command = getattr(args, 'topic_command', None)
    if not topic_command:
        raise CLIError("topics subcommand required")
    return f"topics_{topic_command}", args


def main():
    """Main CLI entry point."""
    # Build command specifications
    commands = [
        # Test embedding
        CommandSpec("test-embedding", parse_test_embedding),
        
        # Profile commands
        CommandSpec("profile", lambda s: _parse_profile_subcommand(s)),
        
        # Ingest commands
        CommandSpec(
            "ingest", 
            lambda s: _parse_ingest_subcommand(s),
            help_text="ingest <subcommand>\n\nSubcommands:\n  all <file> [model <name>] [batch-size <n>] [profile <name>] - Run all stages\n  stage0 <file> [profile <name>] - Parse and store messages/chunks\n  stage1 [model <name>] [batch-size <n>] [profile <name>] - Generate embeddings\n  stage2 [min-cluster-size <n>] [min-samples <n>] [metric <name>] [cluster-selection-method <name>] [cluster-selection-epsilon <f>] [profile <name>] - Cluster chunk embeddings into L1 topics (HDBSCAN)\n  stage3 [profile <name>] - Create embeddings for clusters and assign topic_l1_id to chunks\n  stage4 [only-unnamed] [rebuild] [profile <name>] - Generate topic names for L1 topics\n  stage5 [min-cluster-size <n>] [min-samples <n>] [metric <name>] [cluster-selection-method <name>] [cluster-selection-epsilon <f>] [profile <name>] - Cluster L1 topics into L2 and generate names\n  clear all [profile <name>] - Clear all stages\n  clear stage0 [profile <name>] - Clear messages\n  clear stage1 [profile <name>] - Clear embeddings\n  clear stage2 [profile <name>] - Clear L1 topics\n  clear stage3 [profile <name>] - Clear topic_l1_id assignments\n  clear stage4 [profile <name>] - Clear L1 topic names\n  clear stage5 [profile <name>] - Clear L2 topics and assignments\n  info [profile <name>] - Show database statistics"
        ),
        
        # Telegram commands
        CommandSpec("telegram", lambda s: _parse_telegram_subcommand(s)),
        
        # Chat
        CommandSpec("chat", parse_chat),
        
        # Bot commands
        CommandSpec("bot", lambda s: _parse_bot_subcommand(s)),
        
        # Config commands
        CommandSpec("config", lambda s: _parse_config_subcommand(s)),
        
        # Topics commands
        CommandSpec("topics", lambda s: _parse_topics_subcommand(s)),
    ]
    
    parser = CommandParser(commands)
    
    try:
        # Parse arguments (skip script name)
        cmd_name, args = parser.parse(sys.argv[1:])
    
    except CLIHelp:
        # Check if it's a specific command help request
        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            help_text = parser.get_help(cmd)
            print(help_text)
        else:
            print(parser.get_help())
        sys.exit(0)
    except CLIError as e:
        # Setup logging early for error messages
        setup_log(LOG_WARNING)  # Use default level for error messages
        syslog2(LOG_ERR, f"Error: {e}")
        sys.exit(1)
    
    # Handle subcommands that need routing
    if cmd_name == "profile":
        cmd_name, args = _parse_profile_subcommand_with_args(args)
    elif cmd_name == "ingest":
        cmd_name, args = _parse_ingest_subcommand_with_args(args)
    elif cmd_name == "telegram":
        cmd_name, args = _parse_telegram_subcommand_with_args(args)
    elif cmd_name == "bot":
        cmd_name, args = _parse_bot_subcommand_with_args(args)
    elif cmd_name == "config":
        cmd_name, args = _parse_config_subcommand_with_args(args)
    elif cmd_name == "topics":
        cmd_name, args = _parse_topics_subcommand_with_args(args)

    # Setup global logging
    syslog_level = LOG_WARNING
    log_level = getattr(args, 'log_level', None)
    if log_level:
        syslog_level = log_level
    
    setup_log(syslog_level)
    
    # Initialize profile manager
    profile_manager = ProfileManager(project_root)
    
    # Ensure default profile exists (except for profile commands)
    if not cmd_name.startswith('profile'):
        default_profile = profile_manager.get_profile_dir('default')
        if not default_profile.exists():
            print("Creating default profile...")
            profile_manager.create_profile('default', set_active=True)
            print()
    
    # Route to appropriate command handler
    try:
        if cmd_name == 'test-embedding':
            cmd_test_embedding(args)
        
        elif cmd_name.startswith('profile_'):
            cmd_profile(args, profile_manager)
        
        elif cmd_name.startswith('ingest_'):
            cmd_ingest(args, profile_manager)
        
        elif cmd_name.startswith('telegram_'):
            cmd_telegram(args, profile_manager)
        
        elif cmd_name == 'chat':
            cmd_chat(args, profile_manager)
        
        elif cmd_name.startswith('bot_'):
            cmd_bot(args, profile_manager)

        elif cmd_name.startswith('config_'):
            cmd_config(args, profile_manager)

        elif cmd_name.startswith('topics_'):
            cmd_topics(args, profile_manager)

        else:
            syslog2(LOG_ERR, f"Unknown command: {cmd_name}")
            sys.exit(1)
    except Exception as e:
        syslog2(LOG_ERR, f"Command execution failed", command=cmd_name, error=str(e))
        sys.exit(1)


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
        print(f"Error: No database found for profile '{profile_name}'")
        sys.exit(1)
        
    db = Database(paths['db_url'])
    
    # For 'list' and 'show' commands, we don't need VectorStore or LLMClient
    # Initialize them only for commands that need clustering/naming
    needs_clustering = args.topic_command in ('cluster-l1', 'cluster-l2', 'name', 'build')
    
    if needs_clustering:
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
            
        # For topic naming, always use verbosity=0 to suppress raw LLM output
        # This ensures clean progress bar output without HTTP request/response spam
        llm_verbosity = 0
        llm_client = LLMClient(model=model_name, verbosity=llm_verbosity)
        clusterer = TopicClusterer(db, vector_store, llm_client)
    else:
        clusterer = None

    # Helper function to get clustering parameters
    def get_clustering_params(prefix):
        return {
            'min_cluster_size': getattr(args, f'{prefix}_min_size', 2),
            'min_samples': getattr(args, f'{prefix}_min_samples', 1),
            'metric': getattr(args, f'{prefix}_metric', 'cosine'),
            'cluster_selection_method': getattr(args, f'{prefix}_method', 'eom'),
            'cluster_selection_epsilon': getattr(args, f'{prefix}_epsilon', 0.0)
        }
    
    # Helper function to validate clustering parameters
    def validate_clustering_params(prefix):
        min_size = getattr(args, f'{prefix}_min_size', 2)
        if min_size < 2:
            print(f"Error: --{prefix}-min-size must be at least 2 (HDBSCAN requirement), got {min_size}")
            sys.exit(1)
    
    # Progress bar callback for naming
    def show_progress(current, total, stage):
        """Display progress bar for topic naming."""
        percentage = int((current / total * 100)) if total > 0 else 0
        bar_width = 30
        filled = int((current / total * bar_width)) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        stage_name = "L1 Topics" if stage == 'l1' else "L2 Topics"
        print(f"\r  {stage_name}: [{bar}] {current}/{total} ({percentage}%)", end="", flush=True)
        if current == total:
            print()  # New line when complete

    if args.topic_command == 'cluster-l1':
        print(f"Running L1 clustering for profile: {profile_name}")
        validate_clustering_params('l1')
        params = get_clustering_params('l1')
        print(f"Parameters: min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}, "
              f"metric={params['metric']}, method={params['cluster_selection_method']}, epsilon={params['cluster_selection_epsilon']}")
        try:
            clusterer.perform_l1_clustering(**params)
            print("L1 clustering complete. Run 'topics cluster-l2' for super-topics.")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.topic_command == 'cluster-l2':
        print(f"Running L2 clustering for profile: {profile_name}")
        # Check if L1 topics exist
        l1_topics = db.get_all_topics_l1()
        if not l1_topics:
            print("Error: No L1 topics found. Run 'topics cluster-l1' first.")
            sys.exit(1)
        validate_clustering_params('l2')
        params = get_clustering_params('l2')
        print(f"Parameters: min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}, "
              f"metric={params['metric']}, method={params['cluster_selection_method']}, epsilon={params['cluster_selection_epsilon']}")
        try:
            clusterer.perform_l2_clustering(**params)
            print("L2 clustering complete. Run 'topics name' to generate topic names.")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
    elif args.topic_command == 'name':
        print(f"Generating topic names for profile: {profile_name}")
        # Check if topics exist
        l1_topics = db.get_all_topics_l1()
        l2_topics = db.get_all_topics_l2()
        if not l1_topics and not l2_topics:
            print("Error: No topics found. Run 'topics cluster-l1' first.")
            sys.exit(1)
        
        only_unnamed = getattr(args, 'only_unnamed', False)
        rebuild = getattr(args, 'rebuild', False)
        target = getattr(args, 'target', 'both')
        
        print("Naming Topics (LLM)...")
        clusterer.name_topics(
            progress_callback=show_progress,
            only_unnamed=only_unnamed,
            rebuild=rebuild,
            target=target
        )
        print("Topic naming complete.")
        
    elif args.topic_command == 'build':
        print(f"Building topics for profile: {profile_name}")
        
        # Validate parameters
        validate_clustering_params('l1')
        validate_clustering_params('l2')
        
        # Get clustering parameters
        l1_params = get_clustering_params('l1')
        l2_params = get_clustering_params('l2')
        
        print(f"L1 parameters: min_cluster_size={l1_params['min_cluster_size']}, min_samples={l1_params['min_samples']}, "
              f"metric={l1_params['metric']}, method={l1_params['cluster_selection_method']}, epsilon={l1_params['cluster_selection_epsilon']}")
        print("1. Running L1 Clustering (Fine-grained)...")
        try:
            clusterer.perform_l1_clustering(**l1_params)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        print(f"L2 parameters: min_cluster_size={l2_params['min_cluster_size']}, min_samples={l2_params['min_samples']}, "
              f"metric={l2_params['metric']}, method={l2_params['cluster_selection_method']}, epsilon={l2_params['cluster_selection_epsilon']}")
        print("2. Running L2 Clustering (Super-topics)...")
        try:
            clusterer.perform_l2_clustering(**l2_params)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        print("3. Naming Topics (LLM)...")
        clusterer.name_topics(progress_callback=show_progress)
        
        print("Topic build complete.")
            
    elif args.topic_command == 'list':
        try:
            l2_topics = db.get_all_topics_l2()
            l1_topics = db.get_all_topics_l1()
            
            if not l1_topics and not l2_topics:
                print("No topics found. Run 'legale ingest stage2' or 'legale topics build' first.")
                return
            
            # Group L1 topics by L2 parent
            l1_by_l2 = {}
            orphans = []
            for t in l1_topics:
                if t.parent_l2_id:
                    if t.parent_l2_id not in l1_by_l2:
                        l1_by_l2[t.parent_l2_id] = []
                    l1_by_l2[t.parent_l2_id].append(t)
                else:
                    orphans.append(t)
            
            # Show L2 topics with their L1 children
            if l2_topics:
                print(f"\n{'ID':<5} {'L2 Topic Title':<40} {'L1 Count':<10} {'Chunks'}")
                print("-" * 75)
                
                for l2 in l2_topics:
                    children = l1_by_l2.get(l2.id, [])
                    chunks_count = sum(c.chunk_count for c in children)
                    title = l2.title or f"Topic L2-{l2.id}"
                    print(f"{l2.id:<5} {title[:38]:<40} {len(children):<10} {chunks_count}")
                    
                    # Show L1 topics under each L2 topic
                    if children:
                        for l1 in children:
                            l1_title = l1.title or f"Topic L1-{l1.id}"
                            print(f"  └─ {l1.id:<3} {l1_title[:35]:<35} {l1.chunk_count:<10} chunks")
            
            # Show orphaned L1 topics
            if orphans:
                if l2_topics:
                    print("\nOrphaned L1 Topics (No Super-Topic):")
                else:
                    print(f"\n{'ID':<5} {'L1 Topic Title':<40} {'Chunks':<10}")
                print("-" * 60)
                for t in orphans:
                    title = t.title or f"Topic L1-{t.id}"
                    print(f"{t.id:<5} {title[:38]:<40} {t.chunk_count:<10}")
            
            # If no L2 topics but L1 topics exist, show all L1 topics
            if not l2_topics and l1_topics:
                print(f"\n{'ID':<5} {'L1 Topic Title':<40} {'Chunks':<10}")
                print("-" * 60)
                for t in l1_topics:
                    title = t.title or f"Topic L1-{t.id}"
                    print(f"{t.id:<5} {title[:38]:<40} {t.chunk_count:<10}")

                
        except Exception as e:
                print(f"Error listing topics: {e}")
                
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
            print(f"Error showing topic: {e}")



def cmd_profile_option(args, profile_manager: ProfileManager):
    """Handle profile option management commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
        print(f"Error: Profile '{profile_name}' does not exist")
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
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "paraphrase-multilingual-mpnet-base-v2"
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
                print("Error: value is required for 'set' action")
                sys.exit(1)
            if args.value not in all_models:
                print(f"Error: Unknown model '{args.value}'")
                print(f"  Available models: {', '.join(all_models)}")
                sys.exit(1)
            config.embedding_model = args.value
            print(f"Embedding model set to: {args.value}")
    
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
                print("Error: value is required for 'set' action")
                sys.exit(1)
            if args.value.lower() not in available_generators:
                print(f"Error: Unknown generator '{args.value}'")
                print(f"  Available generators: {', '.join(available_generators)}")
                sys.exit(1)
            try:
                config.embedding_generator = args.value.lower()
                print(f"Embedding generator set to: {args.value.lower()}")
            except ValueError as e:
                print(f"Error: {e}")
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
                print("Error: value is required for 'set' action")
                sys.exit(1)
            try:
                freq_value = int(args.value)
                if freq_value < 0:
                    print("Error: frequency must be >= 0")
                    sys.exit(1)
                config.response_frequency = freq_value
                print(f"Response frequency set to: {freq_value}")
            except ValueError:
                print(f"Error: frequency must be an integer")
                sys.exit(1)


def cmd_config(args, profile_manager: ProfileManager):
    """Handle configuration commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
         print(f"Error: Profile '{profile_name}' does not exist")
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
             print(f"System prompt updated for profile '{profile_name}'")
        else:
             print(f"Unknown config key: {args.key}")


if __name__ == '__main__':
    main()
