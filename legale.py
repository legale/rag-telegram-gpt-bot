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
from src.lib.syslog2 import *

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
        syslog2(LOG_ERR, "poetry is not installed")
        syslog2(LOG_NOTICE, "please install poetry first: curl -sSL https://install.python-poetry.org | python3 -")
        syslog2(LOG_NOTICE, "or visit: https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Re-execute this script with poetry run
    cmd = ['poetry', 'run', 'python'] + sys.argv
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        syslog2(LOG_ERR, "failed to run with poetry", error=str(e))
        sys.exit(1)

# Now we're in the virtualenv, continue with normal imports
import warnings
from typing import Optional
from dotenv import load_dotenv, set_key, find_dotenv
from src.lib.syslog2 import *
from src.lib.argparse2 import parse, cmd_parse, gen_help, split_args, matches

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
        syslog2(LOG_NOTICE, "active profile set", profile=profile_name)
    
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
            syslog2(LOG_WARNING, "profile already exists", profile=profile_name, path=str(profile_dir))
            return profile_dir
        
        # Create profile directory structure
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "chroma_db").mkdir(exist_ok=True)
        
        syslog2(LOG_NOTICE, "profile created", profile=profile_name, path=str(profile_dir), database=str(profile_dir / 'legale_bot.db'), vector_store=str(profile_dir / 'chroma_db'))
        
        if set_active:
            self.set_current_profile(profile_name)
        
        return profile_dir
    
    def list_profiles(self):
        """List all available profiles."""
        if not self.profiles_dir.exists():
            syslog2(LOG_NOTICE, "no profiles directory found, create your first profile with: legale profile create <name>")
            return
        
        current = self.get_current_profile()
        profiles = [p.name for p in self.profiles_dir.iterdir() if p.is_dir()]
        
        if not profiles:
            syslog2(LOG_NOTICE, "no profiles found, create your first profile with: legale profile create <name>")
            return
        
        syslog2(LOG_NOTICE, "available profiles")
        for profile in sorted(profiles):
            marker = " (active)" if profile == current else ""
            profile_dir = self.profiles_dir / profile
            db_path = profile_dir / "legale_bot.db"
            db_exists = "OK" if db_path.exists() else "MISSING"
            syslog2(LOG_NOTICE, "profile", name=profile, db_exists=db_exists, marker=marker)

        syslog2(LOG_NOTICE, "active profile", profile=current)
    
    def delete_profile(self, profile_name: str, force: bool = False):
        """Delete a profile and all its data."""
        if profile_name == self.get_current_profile() and not force:
            syslog2(LOG_WARNING, "cannot delete active profile", profile=profile_name, message="set another profile first or use force flag")
            return
        
        profile_dir = self.get_profile_dir(profile_name)
        
        if not profile_dir.exists():
            syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
            return
        
        if not force:
            response = input(f"Delete profile '{profile_name}' and all its data? [y/N]: ")
            if response.lower() != 'y':
                syslog2(LOG_NOTICE, "operation cancelled")
                return
        
        import shutil
        shutil.rmtree(profile_dir)
        syslog2(LOG_NOTICE, "profile deleted", profile=profile_name)
    
    def get_profile_paths(self, profile_name: Optional[str] = None) -> dict:
        """Get all relevant paths for a profile."""
        profile_dir = self.get_profile_dir(profile_name)
        
        return {
            'profile_dir': profile_dir,
            'db_path': profile_dir / 'legale_bot.db',
            'db_url': f"sqlite:///{profile_dir / 'legale_bot.db'}",
            'vector_db_path': profile_dir / 'chroma_db',
            'session_file': self.project_root / 'telegram_session.session',  # Shared session for all profiles
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
        syslog2(LOG_NOTICE, "current profile", profile=current_profile)
    
    elif args.profile_command == 'set':
        # Check if profile exists
        profile_dir = profile_manager.get_profile_dir(args.name)
        if not profile_dir.exists():
            syslog2(LOG_ERR, "profile does not exist", profile=args.name)
            syslog2(LOG_NOTICE, "create it with", command=f"legale profile create {args.name}")
            sys.exit(1)
        
        profile_manager.set_current_profile(args.name)
    
    elif args.profile_command == 'delete':
        profile_manager.delete_profile(args.name, force=getattr(args, 'force', False))
    
    elif args.profile_command == 'info':
        profile_name = args.name if args.name else profile_manager.get_current_profile()
        paths = profile_manager.get_profile_paths(profile_name)
        
        db_exists = 'exists' if paths['db_path'].exists() else 'not created'
        vec_exists = 'exists' if paths['vector_db_path'].exists() else 'not created'
        sess_exists = 'exists' if paths['session_file'].exists() else 'not created'
        syslog2(LOG_NOTICE, "profile info", profile=profile_name, directory=str(paths['profile_dir']), database=str(paths['db_path']), db_status=db_exists, vector_db=str(paths['vector_db_path']), vec_status=vec_exists, session=str(paths['session_file']), sess_status=sess_exists, note="session is shared for all profiles")
    
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
    
    syslog2(LOG_NOTICE, "using profile", profile=profile_name, database=str(paths['db_path']), vector_store=str(paths['vector_db_path']))
    
    # Route to appropriate subcommand
    ingest_command = getattr(args, 'ingest_command', None)
    
    # Create pipeline with profile-specific paths
    pipeline = IngestionPipeline(
        db_url=paths['db_url'],
        vector_db_path=str(paths['vector_db_path']),
        profile_dir=str(paths['profile_dir'])
    )
    
    # For 'info' command, use the new get_ingest_info() method
    if ingest_command == 'info':
        info_output = pipeline.get_ingest_info()
        print(info_output)
        return
    
    if ingest_command == 'all':
        if not args.file:
            syslog2(LOG_ERR, "file path is required for ingest all")
            sys.exit(1)
        model = getattr(args, 'model', None)
        batch_size = getattr(args, 'batch_size', 128)
        # Only stages 0-3 are run: messages, chunks, embeddings, vector sync
        pipeline.run_all(args.file, model=model, batch_size=batch_size)
        
    elif ingest_command == 'stage0':
        if not args.file:
            syslog2(LOG_ERR, "file path is required for ingest stage0")
            sys.exit(1)
        syslog2(LOG_NOTICE, "running stage0: parse and store")
        pipeline.run_stage0(args.file)
        syslog2(LOG_NOTICE, "stage0 complete")
        
    elif ingest_command == 'stage1':
        # Check if there are messages in database
        from src.storage.db import Database
        db = Database(paths['db_url'])
        message_count = db.count_messages()
        if message_count == 0:
            syslog2(LOG_ERR, "no messages found in database, run ingest stage0 first")
            sys.exit(1)
        
        syslog2(LOG_NOTICE, "running stage1: create and store chunks")
        pipeline.run_stage1()
        syslog2(LOG_NOTICE, "stage1 complete")
        
    elif ingest_command == 'stage2':
        # Check if there are chunks in database
        from src.storage.db import Database
        db = Database(paths['db_url'])
        chunk_count = db.count_chunks()
        if chunk_count == 0:
            syslog2(LOG_ERR, "no chunks found in database, run ingest stage1 first")
            sys.exit(1)
        
        # Get optional parameters for embedding generation
        model = getattr(args, 'model', None)
        batch_size = getattr(args, 'batch_size', 128)
        
        syslog2(LOG_NOTICE, "running stage2: generate embeddings")
        pipeline.run_stage2(model=model, batch_size=batch_size)
        syslog2(LOG_NOTICE, "stage2 complete")
        
    elif ingest_command == 'stage3':
        # Check if there are embeddings in SQLite
        from src.storage.db import Database, ChunkModel
        db = Database(paths['db_url'])
        session = db.get_session()
        try:
            chunks_with_embeddings = session.query(ChunkModel).filter(
                ChunkModel.embedding_json.isnot(None)
            ).count()
            if chunks_with_embeddings == 0:
                syslog2(LOG_ERR, "no embeddings found in sqlite, run ingest stage2 first")
                sys.exit(1)
        finally:
            session.close()
        
        syslog2(LOG_NOTICE, "running stage3: sync chunks to vector database")
        pipeline.run_stage3()
        syslog2(LOG_NOTICE, "stage3 complete")
        
    elif ingest_command == 'clear_all':
        syslog2(LOG_NOTICE, "clearing all stages")
        pipeline.clear_all()
        syslog2(LOG_NOTICE, "all stages cleared")
        
    elif ingest_command == 'clear_stage0':
        syslog2(LOG_NOTICE, "clearing stage0: messages")
        deleted = pipeline.clear_stage0()
        syslog2(LOG_NOTICE, "stage0 cleared", messages_deleted=deleted)
        
    elif ingest_command == 'clear_stage1':
        syslog2(LOG_NOTICE, "clearing stage1: chunks")
        deleted = pipeline.clear_stage1()
        syslog2(LOG_NOTICE, "stage1 cleared", chunks_deleted=deleted)
        
    elif ingest_command == 'clear_stage2':
        syslog2(LOG_NOTICE, "clearing stage2: chunk embeddings (SQLite)")
        updated = pipeline.clear_stage2()
        syslog2(LOG_NOTICE, "stage2 cleared", embeddings_cleared=updated)
        
    elif ingest_command == 'clear_stage3':
        syslog2(LOG_NOTICE, "clearing stage3: vector_db chunks")
        removed = pipeline.clear_stage3()
        syslog2(LOG_NOTICE, "stage3 cleared", vectors_removed=removed)
        
    else:
        # Should not happen due to routing logic, but handle gracefully
        syslog2(LOG_ERR, "unknown ingest command")
        sys.exit(1)


def cmd_telegram(args, profile_manager: ProfileManager):
    """Handle Telegram data fetching commands."""
    from src.ingestion.telegram import TelegramFetcher
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    if not API_ID or not API_HASH:
        syslog2(LOG_ERR, "telegram_api_id and telegram_api_hash must be set in .env file")
        sys.exit(1)
    
    try:
        API_ID = int(API_ID)
    except ValueError:
        syslog2(LOG_ERR, "telegram_api_id must be an integer")
        sys.exit(1)
    
    # Get profile-specific paths (session file is shared for all profiles)
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    paths = profile_manager.get_profile_paths(profile_name)
    
    # Ensure profile directory exists
    paths['profile_dir'].mkdir(parents=True, exist_ok=True)
    
    # Session file is in project root, shared for all profiles
    session_name = str(paths['session_file'].with_suffix(''))  # Remove .session extension
    
    syslog2(LOG_NOTICE, "using profile", profile=profile_name, session_file=str(paths['session_file']), note="shared session for all profiles")
    
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
        with fetcher.client:
            chat = fetcher._find_chat(args.target)
            if not chat:
                syslog2(LOG_ERR, "chat not found", name=args.target)
                syslog2(LOG_ERR, "chat not found", target=args.target)
                sys.exit(1)
            
            chat_id = abs(chat.id)  # Use absolute value for negative IDs
            output_dir = str(paths['profile_dir'])
            dump_file = os.path.join(output_dir, f"telegram_dump_{chat_id}.json")
        
        # Dump chat
        fetcher.dump_chat(args.target, limit=args.limit, output_file=output_dir)
        
        # Check if dump file was created
        if not os.path.exists(dump_file):
            syslog2(LOG_ERR, "failed to create dump file", file=dump_file)
            sys.exit(1)
        
        syslog2(LOG_NOTICE, "starting ingestion", dump_file=dump_file)
        
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
        syslog2(LOG_ERR, "no database found for profile", profile=profile_name)
        syslog2(LOG_NOTICE, "create one by ingesting data first: legale ingest <file>")
        sys.exit(1)
    
    syslog2(LOG_NOTICE, "using profile", profile=profile_name, database=str(paths['db_path']))
    
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
    if hasattr(args, 'retrieval_type') and args.retrieval_type:
        cli_args.extend(['--retrieval-type', args.retrieval_type])
    
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
    
    syslog2(LOG_NOTICE, "using profile", profile=profile_name)
    
    if args.bot_command == 'register':
        token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            syslog2(LOG_ERR, "telegram_bot_token must be set in .env or passed via --token")
            sys.exit(1)
        
        register_webhook(args.url, token)
    
    elif args.bot_command == 'delete':
        token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            syslog2(LOG_ERR, "telegram_bot_token must be set in .env or passed via --token")
            sys.exit(1)
        
        delete_webhook(token)
    
    elif args.bot_command == 'run':
        syslog2(LOG_NOTICE, "database", path=str(paths['db_path']))
        syslog2(LOG_NOTICE, "vector store", path=str(paths['vector_db_path']))
        
        # Convert log_level to string if it's a number, handle both string and numeric log levels
        log_level = getattr(args, 'log_level', None)
        if log_level:
            # Convert to string if it's a number
            if not isinstance(log_level, str):
                log_level = str(log_level)
        
        debug_rag = getattr(args, 'debug_rag', False)
        run_server(args.host, args.port, log_level=log_level, debug_rag=debug_rag, args=args)
    
    elif args.bot_command == 'daemon':
        syslog2(LOG_NOTICE, "database", path=str(paths['db_path']))
        syslog2(LOG_NOTICE, "vector store", path=str(paths['vector_db_path']))
        
        run_daemon(args.host, args.port, args=args)


def cmd_test_embedding(args):
    """Test embedding generation for input text."""
    # NOTE: We intentionally allow online mode here for initial model download
    # After first download, the model will be cached and work offline
    
    from src.core.embedding import LocalEmbeddingClient
    import time
    
    model = getattr(args, 'model', 'ai-sage/Giga-Embeddings-instruct')
    text = getattr(args, 'text', '')
    
    syslog2(LOG_NOTICE, "testing embedding generation", model=model, text=text)
    
    try:
        # Initialize client and warmup model (exclude from timing)
        syslog2(LOG_NOTICE, "loading model")
        client = LocalEmbeddingClient(model=model)
        
        # Warmup: generate embedding once to load model into memory
        _ = client.get_embedding("warmup")
        syslog2(LOG_NOTICE, "model loaded")
        
        # Now measure only embedding generation time
        start_time = time.time()
        emb = client.get_embedding(text)
        duration_ms = (time.time() - start_time) * 1000
        
        syslog2(LOG_NOTICE, "embedding generation success", duration_ms=int(duration_ms), dimensions=len(emb), first_5_values=emb[:5])
    except Exception as e:
        setup_log(LOG_NOTICE)
        syslog2(LOG_ERR, f"Error: {e}")
        sys.exit(1)


# ===== PARSER FUNCTIONS =====

def parse_test_embedding(opts: dict, args: list) -> dict:
    """Parse test-embedding command."""
    if not args:
        raise ValueError("text required for test-embedding")
    text = args[0]
    model = opts.get("model") or "ai-sage/Giga-Embeddings-instruct"
    return {"text": text, "model": model}


def parse_profile_list(opts: dict, args: list) -> dict:
    """Parse profile list command."""
    return {"profile": opts.get("profile")}


def parse_profile_create(opts: dict, args: list) -> dict:
    """Parse profile create command."""
    if not args:
        raise ValueError("profile name required for profile create")
    name = args[0]
    set_active = opts.get("set-active", False)
    return {"name": name, "set_active": set_active, "profile": opts.get("profile")}


def parse_profile_get(opts: dict, args: list) -> dict:
    """Parse profile get command - returns current profile name."""
    # No arguments needed, just return empty dict
    return {"profile": opts.get("profile")}


def parse_profile_set(opts: dict, args: list) -> dict:
    """Parse profile set command."""
    if not args:
        raise ValueError("profile name required for profile set")
    name = args[0]
    return {"name": name, "profile": opts.get("profile")}


def parse_profile_delete(opts: dict, args: list) -> dict:
    """Parse profile delete command."""
    if not args:
        raise ValueError("profile name required for profile delete")
    name = args[0]
    force = opts.get("force", False)
    return {"name": name, "force": force, "profile": opts.get("profile")}


def parse_profile_info(opts: dict, args: list) -> dict:
    """Parse profile info command."""
    name = args[0] if args and not args[0].startswith("-") else None
    profile = opts.get("profile")
    return {"name": name, "profile": profile}


def parse_profile_option(opts: dict, args: list) -> dict:
    """Parse profile option command."""
    if len(args) < 2:
        raise ValueError("option and action required for profile option")
    option = args[0]
    action = args[1]
    value = args[2] if len(args) > 2 and not args[2].startswith("-") else None
    profile = opts.get("profile")
    return {"option": option, "action": action, "value": value, "profile": profile}


def parse_ingest_all(opts: dict, args: list) -> dict:
    """Parse ingest all command."""
    if not args:
        raise ValueError("file path required for ingest all")
    file = args[0]
    model = opts.get("model")
    batch_size = int(opts.get("batch-size", 128)) if opts.get("batch-size") else 128
    profile = opts.get("profile")
    return {"file": file, "model": model, "batch_size": batch_size, "profile": profile, "ingest_command": "all"}


def parse_ingest_stage0(opts: dict, args: list) -> dict:
    """Parse ingest stage0 command."""
    if not args:
        raise ValueError("file path required for ingest stage0")
    file = args[0]
    profile = opts.get("profile")
    return {"file": file, "profile": profile, "ingest_command": "stage0"}


def parse_ingest_stage1(opts: dict, args: list) -> dict:
    """Parse ingest stage1 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "stage1"}


def parse_ingest_stage2(opts: dict, args: list) -> dict:
    """Parse ingest stage2 command."""
    model = opts.get("model")
    batch_size = int(opts.get("batch-size", 128)) if opts.get("batch-size") else 128
    profile = opts.get("profile")
    return {
        "model": model,
        "batch_size": batch_size,
        "profile": profile,
        "ingest_command": "stage2"
    }


def parse_ingest_stage3(opts: dict, args: list) -> dict:
    """Parse ingest stage3 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "stage3"}






def parse_ingest_stage8(opts: dict, args: list) -> dict:
    """Parse ingest stage8 command."""
    rebuild = opts.get("rebuild", False)
    profile = opts.get("profile")
    return {"rebuild": rebuild, "profile": profile, "ingest_command": "stage8"}


def parse_ingest_stage9(opts: dict, args: list) -> dict:
    """Parse ingest stage9 command."""
    rebuild = opts.get("rebuild", False)
    profile = opts.get("profile")
    return {"rebuild": rebuild, "profile": profile, "ingest_command": "stage9"}


def parse_ingest_clear_all(opts: dict, args: list) -> dict:
    """Parse ingest clear all command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "clear_all"}


def parse_ingest_clear_stage0(opts: dict, args: list) -> dict:
    """Parse ingest clear stage0 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "clear_stage0"}


def parse_ingest_clear_stage1(opts: dict, args: list) -> dict:
    """Parse ingest clear stage1 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "clear_stage1"}


def parse_ingest_clear_stage2(opts: dict, args: list) -> dict:
    """Parse ingest clear stage2 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "clear_stage2"}


def parse_ingest_clear_stage3(opts: dict, args: list) -> dict:
    """Parse ingest clear stage3 command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "clear_stage3"}






def parse_ingest_info(opts: dict, args: list) -> dict:
    """Parse ingest info command."""
    profile = opts.get("profile")
    return {"profile": profile, "ingest_command": "info"}


def parse_ingest(opts: dict, args: list) -> dict:
    """Parse ingest command without subcommand (treats as 'all')."""
    if not args:
        raise ValueError("ingest subcommand required (all, stage0-9, clear all/stage0-9, info) or file path")
    
    # Check if first argument is a subcommand
    first = args[0].lower()
    if first in ("all", "stage0", "stage1", "stage2", "stage3", "clear", "info"):
        raise ValueError(f"ingest subcommand '{first}' requires explicit subcommand syntax")
    
    # Treat as 'all' with file path
    file = args[0]
    model = opts.get("model")
    batch_size = int(opts.get("batch-size", 128)) if opts.get("batch-size") else 128
    profile = opts.get("profile")
    return {"file": file, "model": model, "batch_size": batch_size, "profile": profile, "ingest_command": "all"}




def parse_telegram_list(opts: dict, args: list) -> dict:
    """Parse telegram list command."""
    return {"profile": opts.get("profile"), "telegram_command": "list"}


def parse_telegram_members(opts: dict, args: list) -> dict:
    """Parse telegram members command."""
    if not args:
        raise ValueError("target required for telegram members")
    target = args[0]
    profile = opts.get("profile")
    return {"target": target, "profile": profile, "telegram_command": "members"}


def parse_telegram_dump(opts: dict, args: list) -> dict:
    """Parse telegram dump command."""
    if not args:
        raise ValueError("target required for telegram dump")
    target = args[0]
    limit = int(opts.get("limit", 1000)) if opts.get("limit") else 1000
    output = opts.get("output")
    profile = opts.get("profile")
    return {"target": target, "limit": limit, "output": output, "profile": profile, "telegram_command": "dump"}


def parse_telegram_ingest_all(opts: dict, args: list) -> dict:
    """Parse telegram ingest all command."""
    if not args:
        raise ValueError("target required for telegram ingest all")
    target = args[0]
    limit = int(opts.get("limit", 1000)) if opts.get("limit") else 1000
    model = opts.get("model")
    batch_size = int(opts.get("batch-size", 128)) if opts.get("batch-size") else 128
    profile = opts.get("profile")
    return {"target": target, "limit": limit, "model": model, "batch_size": batch_size, "profile": profile, "telegram_command": "ingest_all"}


def parse_chat(opts: dict, args: list) -> dict:
    """Parse chat command."""
    chunks = int(opts.get("chunks")) if opts.get("chunks") else None
    debug_rag = opts.get("debug-rag", False)
    retrieval_type = opts.get("retrieval-type", "hybrid")
    if retrieval_type not in ["hybrid", "fts_only", "vector_only"]:
        raise ValueError(f"invalid retrieval-type: {retrieval_type}, must be one of: hybrid, fts_only, vector_only")
    profile = opts.get("profile")
    return {"chunks": chunks, "debug_rag": debug_rag, "retrieval_type": retrieval_type, "profile": profile}


def parse_bot_register(opts: dict, args: list) -> dict:
    """Parse bot register command."""
    url = opts.get("url")
    if not url:
        raise ValueError("url required for bot register")
    token = opts.get("token")
    profile = opts.get("profile")
    return {"url": url, "token": token, "profile": profile, "bot_command": "register"}


def parse_bot_delete(opts: dict, args: list) -> dict:
    """Parse bot delete command."""
    token = opts.get("token")
    profile = opts.get("profile")
    return {"token": token, "profile": profile, "bot_command": "delete"}


def parse_bot_run(opts: dict, args: list) -> dict:
    """Parse bot run command."""
    host = opts.get("host") or "127.0.0.1"
    port = int(opts.get("port", 8000)) if opts.get("port") else 8000
    profile = opts.get("profile")
    debug_rag = opts.get("debug-rag", False)
    retrieval_type = opts.get("retrieval-type", "hybrid")
    if retrieval_type not in ["hybrid", "fts_only", "vector_only"]:
        raise ValueError(f"invalid retrieval-type: {retrieval_type}, must be one of: hybrid, fts_only, vector_only")
    return {"host": host, "port": port, "profile": profile, "debug_rag": debug_rag, "retrieval_type": retrieval_type, "bot_command": "run"}


def parse_bot_daemon(opts: dict, args: list) -> dict:
    """Parse bot daemon command."""
    host = opts.get("host") or "127.0.0.1"
    port = int(opts.get("port", 8000)) if opts.get("port") else 8000
    profile = opts.get("profile")
    return {"host": host, "port": port, "profile": profile, "bot_command": "daemon"}


def parse_config_get(opts: dict, args: list) -> dict:
    """Parse config get command."""
    if not args:
        raise ValueError("key required for config get")
    key = args[0]
    profile = opts.get("profile")
    return {"key": key, "profile": profile, "config_command": "get"}


def parse_config_set(opts: dict, args: list) -> dict:
    """Parse config set command."""
    if len(args) < 2:
        raise ValueError("key and value required for config set")
    key = args[0]
    value = args[1]
    profile = opts.get("profile")
    return {"key": key, "value": value, "profile": profile, "config_command": "set"}




# Subcommand parsers
def _parse_profile_subcommand(opts: dict, args: list) -> dict:
    """Parse profile subcommand."""
    if not args:
        raise ValueError("profile subcommand required (list, create, get, set, delete, info, option)")
    subcmd = args[0].lower()
    remaining_args = args[1:]
    
    result = None
    if subcmd == "list":
        result = parse_profile_list(opts, remaining_args)
        result["profile_command"] = "list"
    elif subcmd == "create":
        result = parse_profile_create(opts, remaining_args)
        result["profile_command"] = "create"
    elif subcmd == "get":
        result = parse_profile_get(opts, remaining_args)
        result["profile_command"] = "get"
    elif subcmd == "set":
        result = parse_profile_set(opts, remaining_args)
        result["profile_command"] = "set"
    elif subcmd == "delete":
        result = parse_profile_delete(opts, remaining_args)
        result["profile_command"] = "delete"
    elif subcmd == "info":
        result = parse_profile_info(opts, remaining_args)
        result["profile_command"] = "info"
    elif subcmd == "option":
        result = parse_profile_option(opts, remaining_args)
        result["profile_command"] = "option"
    else:
        raise ValueError(f"unknown profile subcommand: {subcmd}")
    
    return result


def _parse_profile_subcommand_with_args(args):
    """Extract profile subcommand from args."""
    profile_command = getattr(args, 'profile_command', None)
    if not profile_command:
        raise ValueError("profile subcommand required")
    return f"profile_{profile_command}", args


def _parse_ingest_subcommand(opts: dict, args: list) -> dict:
    """Parse ingest subcommand."""
    if not args:
        # Show help if no arguments - return help command
        return {"ingest_command": "help"}
    
    subcmd = args[0].lower()
    remaining_args = args[1:]
    
    if subcmd == "all":
        return parse_ingest_all(opts, remaining_args)
    elif subcmd == "stage0":
        return parse_ingest_stage0(opts, remaining_args)
    elif subcmd == "stage1":
        return parse_ingest_stage1(opts, remaining_args)
    elif subcmd == "stage2":
        return parse_ingest_stage2(opts, remaining_args)
    elif subcmd == "stage3":
        return parse_ingest_stage3(opts, remaining_args)
    elif subcmd == "stage8":
        return parse_ingest_stage8(opts, remaining_args)
    elif subcmd == "stage9":
        return parse_ingest_stage9(opts, remaining_args)
    elif subcmd == "clear":
        # Parse clear subcommand
        if not remaining_args:
            raise ValueError("clear subcommand required (all, stage0-9)")
        clear_subcmd = remaining_args[0].lower()
        clear_remaining = remaining_args[1:]
        if clear_subcmd == "all":
            return parse_ingest_clear_all(opts, clear_remaining)
        elif clear_subcmd == "stage0":
            return parse_ingest_clear_stage0(opts, clear_remaining)
        elif clear_subcmd == "stage1":
            return parse_ingest_clear_stage1(opts, clear_remaining)
        elif clear_subcmd == "stage2":
            return parse_ingest_clear_stage2(opts, clear_remaining)
        elif clear_subcmd == "stage3":
            return parse_ingest_clear_stage3(opts, clear_remaining)
        elif clear_subcmd == "stage8":
            return parse_ingest_clear_stage8(opts, clear_remaining)
        elif clear_subcmd == "stage9":
            return parse_ingest_clear_stage9(opts, clear_remaining)
        else:
            raise ValueError(f"unknown clear subcommand: {clear_subcmd}. Use: all, stage0, stage1, stage2, stage3")
    elif subcmd == "info":
        return parse_ingest_info(opts, remaining_args)
    else:
        raise ValueError(f"unknown ingest subcommand: {subcmd}. Use: all, stage0-9, clear all/stage0-9, info")


def _parse_ingest_subcommand_with_args(args):
    """Extract ingest subcommand from args."""
    ingest_command = getattr(args, 'ingest_command', 'all')
    if ingest_command == "help":
        raise ValueError("ingest subcommand required (all, stage0-9, clear all/stage0-9, info)")
    return f"ingest_{ingest_command}", args


def _parse_telegram_subcommand(opts: dict, args: list) -> dict:
    """Parse telegram subcommand."""
    if not args:
        raise ValueError("telegram subcommand required (list, members, dump, ingest)")
    subcmd = args[0].lower()
    remaining_args = args[1:]
    
    result = None
    if subcmd == "list":
        result = parse_telegram_list(opts, remaining_args)
    elif subcmd == "members":
        result = parse_telegram_members(opts, remaining_args)
    elif subcmd == "dump":
        result = parse_telegram_dump(opts, remaining_args)
    elif subcmd == "ingest":
        # Parse ingest subcommand
        if not remaining_args:
            raise ValueError("telegram ingest subcommand required (all)")
        ingest_subcmd = remaining_args[0].lower()
        ingest_remaining = remaining_args[1:]
        if ingest_subcmd == "all":
            result = parse_telegram_ingest_all(opts, ingest_remaining)
        else:
            raise ValueError(f"unknown telegram ingest subcommand: {ingest_subcmd}")
    else:
        raise ValueError(f"unknown telegram subcommand: {subcmd}")
    
    # telegram_command already set by parsers
    return result


def _parse_telegram_subcommand_with_args(args):
    """Extract telegram subcommand from args."""
    telegram_command = getattr(args, 'telegram_command', None)
    if not telegram_command:
        raise ValueError("telegram subcommand required")
    return f"telegram_{telegram_command}", args


def _parse_bot_subcommand(opts: dict, args: list) -> dict:
    """Parse bot subcommand."""
    if not args:
        raise ValueError("bot subcommand required (register, delete, run, daemon)")
    subcmd = args[0].lower()
    remaining_args = args[1:]
    
    result = None
    if subcmd == "register":
        result = parse_bot_register(opts, remaining_args)
    elif subcmd == "delete":
        result = parse_bot_delete(opts, remaining_args)
    elif subcmd == "run":
        result = parse_bot_run(opts, remaining_args)
    elif subcmd == "daemon":
        result = parse_bot_daemon(opts, remaining_args)
    else:
        raise ValueError(f"unknown bot subcommand: {subcmd}")
    
    # bot_command already set by parsers
    return result


def _parse_bot_subcommand_with_args(args):
    """Extract bot subcommand from args."""
    bot_command = getattr(args, 'bot_command', None)
    if not bot_command:
        raise ValueError("bot subcommand required")
    return f"bot_{bot_command}", args


def _parse_config_subcommand(opts: dict, args: list) -> dict:
    """Parse config subcommand."""
    if not args:
        raise ValueError("config subcommand required (get, set)")
    subcmd = args[0].lower()
    remaining_args = args[1:]
    
    result = None
    if subcmd == "get":
        result = parse_config_get(opts, remaining_args)
    elif subcmd == "set":
        result = parse_config_set(opts, remaining_args)
    else:
        raise ValueError(f"unknown config subcommand: {subcmd}")
    
    # config_command already set by parsers
    return result


def _parse_config_subcommand_with_args(args):
    """Extract config subcommand from args."""
    config_command = getattr(args, 'config_command', None)
    if not config_command:
        raise ValueError("config subcommand required")
    return f"config_{config_command}", args




def main():
    """Main CLI entry point."""

    # Setup logging early for error messages
    setup_log(LOG_NOTICE)  # Use default level for error messages    
    
    # Build opt_table for global options
    opt_table = {
        "V": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "log-level": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "h": {"desc": "Show help"},
        "help": {"desc": "Show help"},
        "v": {"desc": "Show version"},
        "version": {"desc": "Show version"},
    }
    
    # Build cmd_table for commands
    cmd_table = {
        "test-embedding": {"desc": "Test embedding generation"},
        "profile": {"desc": "Profile management"},
        "ingest": {"desc": "Data ingestion"},
        "telegram": {"desc": "Telegram data fetching"},
        "chat": {"desc": "Interactive chat"},
        "bot": {"desc": "Telegram bot webhook"},
        "config": {"desc": "Configuration management"},
    }
    
    try:
        # Parse arguments (skip script name)
        opts, cmd, args = cmd_parse(sys.argv[1:], opt_table)
        
        # Handle help and version flags
        if opts.get("h", False) or opts.get("help", False) or cmd == "help":
            help_text = gen_help("legale", opt_table, cmd_table)
            syslog2(LOG_NOTICE, "help", help_text=help_text)
            sys.exit(0)
        
        if opts.get("v", False) or opts.get("version", False):
            print("legale-bot version 1.0")
            sys.exit(0)
        
        # Handle log level from -V option
        log_level = opts.get("V") or opts.get("log-level")
        
        # Parse command-specific arguments
        if cmd == "test-embedding":
            result = parse_test_embedding(opts, args)
        elif cmd == "profile":
            result = _parse_profile_subcommand(opts, args)
        elif cmd == "ingest":
            result = _parse_ingest_subcommand(opts, args)
        elif cmd == "telegram":
            result = _parse_telegram_subcommand(opts, args)
        elif cmd == "chat":
            result = parse_chat(opts, args)
        elif cmd == "bot":
            result = _parse_bot_subcommand(opts, args)
        elif cmd == "config":
            result = _parse_config_subcommand(opts, args)
        else:
            raise ValueError(f"Unknown command: {cmd}")
        
        # Add log_level to result if present
        if log_level:
            result["log_level"] = log_level
        
        # Convert result dict to SimpleNamespace for compatibility
        from types import SimpleNamespace
        args_obj = SimpleNamespace(**result)
        syslog2(LOG_NOTICE, "args:", **result)
        
        # Handle subcommands that need routing
        if cmd == "profile":
            cmd_name, args_obj = _parse_profile_subcommand_with_args(args_obj)
        elif cmd == "ingest":
            cmd_name, args_obj = _parse_ingest_subcommand_with_args(args_obj)
        elif cmd == "telegram":
            cmd_name, args_obj = _parse_telegram_subcommand_with_args(args_obj)
        elif cmd == "bot":
            cmd_name, args_obj = _parse_bot_subcommand_with_args(args_obj)
        elif cmd == "config":
            cmd_name, args_obj = _parse_config_subcommand_with_args(args_obj)
        else:
            cmd_name = cmd
    
    except ValueError as e:
        syslog2(LOG_ERR, f"Error: {e}")
        sys.exit(1)

    # Setup global logging
    syslog_level = LOG_NOTICE
    log_level = getattr(args_obj, 'log_level', None)
    if log_level:
        # Convert string log level to integer if needed
        # Supports both "LOG_*" and plain names (e.g., "LOG_INFO" or "INFO")
        if isinstance(log_level, str):
            log_level_upper = log_level.upper()
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
            syslog_level = log_level_map.get(log_level_upper, LOG_WARNING)
        else:
            syslog_level = log_level
    
    setup_log(syslog_level)
    
    # Initialize profile manager
    profile_manager = ProfileManager(project_root)
    
    # Ensure default profile exists (except for profile commands)
    if not cmd_name.startswith('profile'):
        default_profile = profile_manager.get_profile_dir('default')
        if not default_profile.exists():
            syslog2(LOG_NOTICE, "creating default profile")
            profile_manager.create_profile('default', set_active=True)
    
    # Route to appropriate command handler
    try:
        if cmd_name == 'test-embedding':
            cmd_test_embedding(args_obj)
        
        elif cmd_name.startswith('profile_'):
            cmd_profile(args_obj, profile_manager)
        
        elif cmd_name.startswith('ingest_'):
            cmd_ingest(args_obj, profile_manager)
        
        elif cmd_name.startswith('telegram_'):
            cmd_telegram(args_obj, profile_manager)
        
        elif cmd_name == 'chat':
            cmd_chat(args_obj, profile_manager)
        
        elif cmd_name.startswith('bot_'):
            cmd_bot(args_obj, profile_manager)

        elif cmd_name.startswith('config_'):
            cmd_config(args_obj, profile_manager)

        else:
            syslog2(LOG_ERR, f"Unknown command: {cmd_name}")
            sys.exit(1)
    except Exception as e:
        syslog2(LOG_ERR, f"Command execution failed", command=cmd_name, error=str(e))
        sys.exit(1)


def cmd_profile_option(args, profile_manager: ProfileManager):
    """Handle profile option management commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
        syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
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
            syslog2(LOG_NOTICE, "available embedding models")
            syslog2(LOG_NOTICE, "openrouter/openai models")
            for model in available_models["openrouter"]:
                marker = " (current)" if model == config.embedding_model else ""
                syslog2(LOG_NOTICE, "model", name=model, marker=marker)
            syslog2(LOG_NOTICE, "local models (sentence-transformers)")
            for model in available_models["local"]:
                marker = " (current)" if model == config.embedding_model else ""
                syslog2(LOG_NOTICE, "model", name=model, marker=marker)
        
        elif action == 'get':
            syslog2(LOG_NOTICE, "current embedding model", model=config.embedding_model)
        
        elif action == 'set':
            if not args.value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            if args.value not in all_models:
                syslog2(LOG_ERR, "unknown model", model=args.value, available=", ".join(all_models))
                sys.exit(1)
            config.embedding_model = args.value
            syslog2(LOG_NOTICE, "embedding model set", model=args.value)
    
    elif option == 'generator':
        if action == 'list':
            syslog2(LOG_NOTICE, "available embedding generators")
            for gen in available_generators:
                marker = " (current)" if gen == config.embedding_generator else ""
                syslog2(LOG_NOTICE, "generator", name=gen, marker=marker)
            syslog2(LOG_NOTICE, "note", openrouter="Use OpenRouter/OpenAI API", local="Use local sentence-transformers (no API key required)")
        
        elif action == 'get':
            syslog2(LOG_NOTICE, "current embedding generator", generator=config.embedding_generator)
        
        elif action == 'set':
            if not args.value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            if args.value.lower() not in available_generators:
                syslog2(LOG_ERR, "unknown generator", generator=args.value, available=", ".join(available_generators))
                sys.exit(1)
            try:
                config.embedding_generator = args.value.lower()
                syslog2(LOG_NOTICE, "embedding generator set", generator=args.value.lower())
            except ValueError as e:
                syslog2(LOG_ERR, "error", error=str(e))
                sys.exit(1)
    
    elif option == 'frequency':
        if action == 'list':
            syslog2(LOG_NOTICE, "response frequency options", option_0="Respond only to mentions", option_1="Respond to every message", option_n="Respond to every N-th message (N > 1)", current=config.response_frequency)
        
        elif action == 'get':
            freq = config.response_frequency
            if freq == 0:
                desc = "Respond only to mentions"
            elif freq == 1:
                desc = "Respond to every message"
            else:
                desc = f"Respond to every {freq}-th message"
            syslog2(LOG_NOTICE, "current response frequency", frequency=freq, description=desc)
        
        elif action == 'set':
            if not args.value:
                syslog2(LOG_ERR, "value is required for set action")
                sys.exit(1)
            try:
                freq_value = int(args.value)
                if freq_value < 0:
                    syslog2(LOG_ERR, "frequency must be >= 0")
                    sys.exit(1)
                config.response_frequency = freq_value
                syslog2(LOG_NOTICE, "response frequency set", frequency=freq_value)
            except ValueError:
                syslog2(LOG_ERR, "frequency must be an integer")
                sys.exit(1)


def cmd_config(args, profile_manager: ProfileManager):
    """Handle configuration commands."""
    from src.bot.config import BotConfig
    
    profile_name = args.profile if args.profile else profile_manager.get_current_profile()
    profile_dir = profile_manager.get_profile_dir(profile_name)
    
    if not profile_dir.exists():
         syslog2(LOG_ERR, "profile does not exist", profile=profile_name)
         sys.exit(1)
         
    config = BotConfig(profile_dir)
    
    if args.config_command == 'get':
        if args.key == 'system_prompt':
            prompt = config.system_prompt
            if not prompt:
                from src.core.prompt import PromptEngine
                prompt = f"(Default)\n{PromptEngine.SYSTEM_PROMPT_TEMPLATE}"
            syslog2(LOG_NOTICE, "system prompt", profile=profile_name, prompt=prompt)
        else:
             syslog2(LOG_ERR, "unknown config key", key=args.key)

    elif args.config_command == 'set':
        if args.key == 'system_prompt':
             config.system_prompt = args.value
             syslog2(LOG_NOTICE, "system prompt updated", profile=profile_name)
        else:
             syslog2(LOG_ERR, "unknown config key", key=args.key)


if __name__ == '__main__':
    main()
