#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.bot.core import LegaleBot
    from src.app.main_cli import create_dispatcher, handle_command
    from src.bot.admin import AdminManager
except ImportError as e:
    # Only try auto-poetry execution if NOT already running under poetry
    # We can check an env var set by poetry, or just check if 'src' is importable after path fix.
    # If it still fails after path fix, it's likely a missing dependency.
    
    # Check if we are already in a poetry environment (simple check)
    if os.environ.get("POETRY_ACTIVE") == "1" or sys.prefix != sys.base_prefix:
        print(f"ImportError running under virtualenv: {e}")
        print("Please ensure you are in the project root and dependencies are installed.")
        sys.exit(1)

    poetry_cmd = "poetry"
    if os.system("which poetry > /dev/null 2>&1") != 0:
        if os.path.exists("/home/ru/.local/bin/poetry"):
            poetry_cmd = "/home/ru/.local/bin/poetry"
    try:
        print("Attempting to restart with poetry...", file=sys.stderr)
        os.execvp(poetry_cmd, [poetry_cmd, "run", "python"] + sys.argv)
    except FileNotFoundError:
        print(f"Error: '{poetry_cmd}' not found.", file=sys.stderr)
        sys.exit(1)

from dotenv import load_dotenv
from src.lib.syslog2 import *
from src.lib.argparse2 import parse

class CLIError(Exception):
    """Raised when CLI arguments are invalid."""
    pass

def _find_and_remove(args: list, token: str) -> bool:
    """Find and remove a token from args list."""
    token_lower = token.lower()
    for i, arg in enumerate(args):
        if arg.lower() == token_lower:
            args.pop(i)
            return True
    return False

def _find_and_remove_next(args: list, token: str) -> str | None:
    """Find a token and return the next value, removing both."""
    token_lower = token.lower()
    for i, arg in enumerate(args):
        if arg.lower() == token_lower:
            args.pop(i)
            if i < len(args):
                return args.pop(i)
            return None
    return None

def parse_int_option(args: list, name: str, default: int | None = None) -> int | None:
    """Parse an integer option from args list."""
    value = _find_and_remove_next(args, name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise CLIError(f"invalid integer value for {name}: {value}")

def parse_flag(args: list, name: str) -> bool:
    """Parse a flag (boolean option) from args list."""
    return _find_and_remove(args, name)

def parse_option(args: list, name: str) -> str | None:
    """Parse an option with value from args list."""
    return _find_and_remove_next(args, name)
from typing import Optional

def main():
    # Load .env from project root
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    
    # Debug: Check if key is loaded
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        syslog2(LOG_WARNING, "api key missing", env_file=dotenv_path)
        # Print first few chars if exists to verify
    
    # Parse arguments using argparse2
    opt_table = {
        "-V": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "--log-level": {"arg": True, "desc": "Set log level", "meta": "LEVEL"},
        "chunks": {"arg": True, "desc": "Number of chunks", "meta": "N"},
        "debug-rag": {"desc": "Enable debug RAG mode"},
        "retrieval-type": {"arg": True, "desc": "Retrieval type", "meta": "TYPE"},
    }
    
    opts, args = parse(sys.argv[1:], opt_table)
    # Handle -V/--log-level (global option)
    log_level_str = opts.get("-V", opts.get("--log-level", LOG_WARNING))
    
    # Parse other options
    chunks = int(opts.get("chunks", 5))
    debug_rag = opts.get("debug-rag", False)
    retrieval_type = opts.get("retrieval-type", "hybrid")
    

    # Configure logging
    syslog_level = LOG_WARNING
    
    if log_level_str is not None:
        # Поддерживаем как строковые уровни (DEBUG, INFO, ...),
        # так и числовые значения (например, 7 для LOG_DEBUG)
        level_map = {
            "DEBUG": LOG_DEBUG,
            "INFO": LOG_INFO,
            "NOTICE": LOG_NOTICE,
            "WARNING": LOG_WARNING,
            "ERR": LOG_ERR,
            "CRIT": LOG_CRIT,
            "ALERT": LOG_ALERT,
        }
        if isinstance(log_level_str, int):
            syslog_level = log_level_str
        else:
            s = str(log_level_str).strip()
            if s.isdigit():
                syslog_level = int(s)
            else:
                syslog_level = level_map.get(s.upper(), LOG_WARNING)

    
    setup_log(syslog_level)

    # Read model from models.txt
    model_name = None
    try:
        with open("models.txt", "r") as f:
            line = f.readline().strip()
            if line:
                model_name = line
    except FileNotFoundError:
        syslog2(LOG_ERR, "models file missing")
        sys.exit(1)

    syslog2(LOG_NOTICE, "cli bot initializing", model=model_name, log_level=syslog_level, chunks=chunks, retrieval_type=retrieval_type)
    
    # Get paths from environment (set by legale.py)
    db_url = os.getenv("DATABASE_URL")
    vector_db_path = os.getenv("VECTOR_DB_PATH")
    profile_dir = os.getenv("PROFILE_DIR")
    
    if not db_url or not vector_db_path:
        syslog2(LOG_ERR, "environment variables missing", vars="DATABASE_URL, VECTOR_DB_PATH")
        print("Please use 'legale chat' command instead of running cli.py directly.")
        return

    bot = _init_bot(db_url, vector_db_path, model_name, syslog_level, debug_rag, profile_dir, retrieval_type)
    if bot is None:
        return

    admin_manager = None
    if profile_dir:
        try:
            admin_manager = AdminManager(Path(profile_dir))
        except Exception:
            admin_manager = None

    dispatcher = create_dispatcher(bot, admin_manager=admin_manager)
    _handle_user_input(bot, dispatcher, chunks, debug_rag)


def _init_bot(db_url: str, vector_db_path: str, model_name: str, syslog_level: int, debug_rag: bool, profile_dir: Optional[str], retrieval_type: str):
    """
    Initialize bot with all dependencies.
    
    Args:
        db_url: Database URL
        vector_db_path: Vector database path
        model_name: Model name
        syslog_level: Logging level
        debug_rag: Whether to enable debug RAG mode
        profile_dir: Profile directory path
        retrieval_type: Retrieval type
        
    Returns:
        LegaleBot instance or None if initialization failed
    """
    try:
        bot = LegaleBot(
            db_url=db_url,
            vector_db_path=vector_db_path,
            model_name=model_name,
            log_level=syslog_level,
            debug_rag=debug_rag,
            profile_dir=profile_dir,
            retrieval_type=retrieval_type
        )
        
        print("Bot ready! Type 'exit' or 'quit' to stop.")
        print("Use /help to see available commands.")
        print("-" * 50)
        
        return bot
    except Exception as e:
        _handle_error("bot initialization", e)
        return None


def _handle_user_input(bot, dispatcher, chunks: int, debug_rag: bool) -> None:
    """
    Handle user input in interactive loop.
    
    Args:
        bot: LegaleBot instance
        dispatcher: CommandDispatcher instance
        chunks: Number of chunks to retrieve
        debug_rag: Whether to show debug RAG info
    """
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Debug RAG mode - show retrieved chunks and prompt
            if debug_rag:
                debug_info = bot.get_rag_debug_info(user_input, n_results=chunks)
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
                print(f"Token count: {debug_info.get('token_count', 'N/A')}")
                print("=" * 70 + "\n")
            
            command_response = handle_command(
                command=user_input,
                dispatcher=dispatcher,
            )
            if command_response is not None:
                print(f"Bot: {command_response}")
                print("-" * 50)
                continue

            response_text = bot.chat(user_input, n_results=chunks)
            print(f"Bot: {response_text}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            _handle_error("chat", e)


def _handle_error(context: str, error: Exception) -> None:
    """
    Handle error with logging.
    
    Args:
        context: Context description (e.g., "bot initialization", "chat")
        error: Exception that occurred
    """
    syslog2(LOG_ERR, f"{context} error", error=str(error))

if __name__ == "__main__":
    main()
