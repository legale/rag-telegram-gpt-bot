#!/usr/bin/env python3
import sys
import os

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.bot.core import LegaleBot
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
from src.core.syslog2 import syslog2, setup_log, LogLevel, LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERR, LOG_CRIT, LOG_ALERT, LOG_NOTICE
from src.core.cli_parser import ArgStream, parse_int_option, parse_flag, parse_option, CLIError

def main():
    # Load .env from project root
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    
    # Debug: Check if key is loaded
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        syslog2(LOG_WARNING, "api key missing", env_file=dotenv_path)
        # Print first few chars if exists to verify
    
    # Parse arguments using custom parser
    stream = ArgStream(sys.argv[1:])
    
    # Handle -V/--log-level (global option)
    log_level_str = parse_option(stream, "-V") or parse_option(stream, "--log-level")
    
    # Parse other options
    chunks = parse_int_option(stream, "--chunks") or 5
    debug_rag = parse_flag(stream, "--debug-rag")
    
    # Count -v flags for verbosity (need to check before stream consumes them)
    verbose = 0
    for arg in sys.argv[1:]:
        if arg == "-v":
            verbose += 1
        elif arg.startswith("-v") and not arg.startswith("-V"):
            # Count v's in -vv, -vvv, etc. (but not -V)
            verbose += len([c for c in arg if c == 'v']) - 1

    # Configure logging
    syslog_level = LOG_WARNING
    
    if log_level_str:
        level_map = {
            "DEBUG": LOG_DEBUG,
            "INFO": LOG_INFO,
            "NOTICE": LOG_NOTICE,
            "WARNING": LOG_WARNING,
            "ERR": LOG_ERR,
            "CRIT": LOG_CRIT,
            "ALERT": LOG_ALERT
        }
        syslog_level = level_map.get(log_level_str.upper(), LOG_WARNING)
        
        # Update verbosity for components that use it (LegaleBot, LLMClient)
        if log_level_str.upper() == 'DEBUG':
            verbose = max(verbose, 2)
        elif log_level_str.upper() == 'INFO':
            verbose = max(verbose, 1)
            
    elif verbose == 1:
        syslog_level = LOG_INFO
    elif verbose >= 2:
        syslog_level = LOG_DEBUG
    
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

    syslog2(LOG_INFO, "cli bot initializing", model=model_name, verbosity=verbose, chunks=chunks)
    
    # Get paths from environment (set by legale.py)
    db_url = os.getenv("DATABASE_URL")
    vector_db_path = os.getenv("VECTOR_DB_PATH")
    profile_dir = os.getenv("PROFILE_DIR")
    
    if not db_url or not vector_db_path:
        syslog2(LOG_ERR, "environment variables missing", vars="DATABASE_URL, VECTOR_DB_PATH")
        print("Please use 'legale chat' command instead of running cli.py directly.")
        return

    try:
        bot = LegaleBot(
            db_url=db_url,
            vector_db_path=vector_db_path,
            model_name=model_name, 
            verbosity=verbose,
            profile_dir=profile_dir
        )
        print("Bot ready! Type 'exit' or 'quit' to stop.")
        print("-" * 50)
    except Exception as e:
        syslog2(LOG_ERR, "bot init failed", error=str(e))
        return

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
            
            response = bot.chat(user_input, n_results=chunks)
            print(f"Bot: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            syslog2(LOG_ERR, "chat error", error=str(e))

if __name__ == "__main__":
    main()
