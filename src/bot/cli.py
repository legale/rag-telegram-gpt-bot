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
from src.core.syslog2 import syslog2, setup_log, LogLevel

def main():
    # Load .env from project root
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    
    # Debug: Check if key is loaded
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        syslog2(LOG_WARNING, "api key missing", env_file=dotenv_path)
        # Print first few chars if exists to verify
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Legale Bot CLI")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (-v, -vv, -vvv)")
    parser.add_argument("--chunks", type=int, default=5, help="Number of context chunks to retrieve (default: 5)")
    parser.add_argument("-V", "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERR", "CRIT", "ALERT"], help="Set logging level")
    args = parser.parse_args()

    # Configure logging
    syslog_level = LOG_WARNING
    
    if args.log_level:
        level_map = {
            "DEBUG": LOG_DEBUG,
            "INFO": LOG_INFO,
            "WARNING": LOG_WARNING,
            "ERR": LOG_ERR,
            "CRIT": LOG_CRIT,
            "ALERT": LOG_ALERT
        }
        syslog_level = level_map.get(args.log_level, LOG_WARNING)
        
        # Update verbosity for components that use it (LegaleBot, LLMClient)
        if args.log_level == 'DEBUG':
            args.verbose = max(args.verbose, 2)
        elif args.log_level == 'INFO':
            args.verbose = max(args.verbose, 1)
            
    elif args.verbose == 1:
        syslog_level = LOG_INFO
    elif args.verbose >= 2:
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

    syslog2(LOG_INFO, "cli bot initializing", model=model_name, verbosity=args.verbose, chunks=args.chunks)
    
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
            verbosity=args.verbose,
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
                
            response = bot.chat(user_input, n_results=args.chunks)
            print(f"Bot: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            syslog2(LOG_ERR, "chat error", error=str(e))

if __name__ == "__main__":
    main()
