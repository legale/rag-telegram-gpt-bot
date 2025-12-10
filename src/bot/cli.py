#!/usr/bin/env python3
import sys
import os
import re

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
from src.core.syslog2 import *
from src.core.cli_parser import ArgStream, parse_int_option, parse_flag, parse_option, CLIError
from typing import Optional, Tuple
from pathlib import Path

def handle_cli_command(command: str, bot, admin_manager, debug_rag: bool) -> Optional[str]:
    """
    Handle CLI commands synchronously (similar to Telegram bot commands).
    
    Args:
        command: Command text (e.g., "/find 2.0 vpn туннель" or "/help")
        bot: LegaleBot instance
        admin_manager: AdminManager instance (can be None)
        debug_rag: Whether to show debug RAG info
        
    Returns:
        Response string if command was handled, None if not a command
    """
    if not command.startswith("/"):
        return None
    
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args_text = parts[1] if len(parts) > 1 else ""
    
    if cmd == "/start":
        return "Привет!\n\nИспользуйте /help для справки."
    
    elif cmd == "/help":
        return (
            "Я анализирую историю чата и отвечаю на вопросы.\n\n"
            "Доступные команды:\n"
            "• /start — приветствие\n"
            "• /help — эта справка\n"
            "• /reset — сбросить контекст разговора\n"
            "• /tokens — показать использование токенов\n"
            "• /model — переключить модель LLM\n"
            "• /find [thr] <запрос> — поиск сообщений по запросу\n\n"
            "Просто напишите свой вопрос!"
        )
    
    elif cmd == "/reset":
        try:
            return bot.reset_context()
        except Exception as e:
            syslog2(LOG_ERR, "reset context failed", error=str(e))
            return "Ошибка при сбросе контекста."
    
    elif cmd == "/tokens":
        try:
            usage = bot.get_token_usage()
            response = (
                f"Использование токенов:\n\n"
                f"Текущее: {usage['current_tokens']:,}\n"
                f"Максимум: {usage['max_tokens']:,}\n"
                f"Использовано: {usage['percentage']}%\n\n"
            )
            if usage["percentage"] > 80:
                response += "Приближаетесь к лимиту! Используйте /reset для сброса."
            elif usage["percentage"] > 50:
                response += "Контекст заполнен наполовину."
            else:
                response += "Достаточно места для разговора."
            return response
        except Exception as e:
            syslog2(LOG_ERR, "get token usage failed", error=str(e))
            return "Ошибка при получении информации о токенах."
    
    elif cmd == "/model":
        try:
            msg = bot.get_model()
            # Save new model to config if admin_manager is available
            if admin_manager:
                try:
                    admin_manager.config.current_model = bot.current_model_name
                except Exception as e:
                    syslog2(LOG_WARNING, "failed to save model to config", error=str(e))
            return msg
        except Exception as e:
            syslog2(LOG_ERR, "get model failed", error=str(e))
            return "Ошибка при переключении модели."
    
    elif cmd == "/find":
        return handle_find_command_cli(args_text, bot, admin_manager, debug_rag)
    
    else:
        return None  # Unknown command


def parse_find_command_args(text: str, admin_manager) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse find command arguments (threshold and query).
    
    Args:
        text: Command text (e.g., "2.0 vpn туннель" or "vpn туннель")
        admin_manager: AdminManager instance for config access
        
    Returns:
        Tuple of (threshold, query) or (None, error_message)
    """
    # Get default threshold from config
    default_threshold = admin_manager.config.cosine_distance_thr if admin_manager else 1.5
    
    if not text or not text.strip():
        return None, (
            f"Использование: /find [thr] <запрос>\n\n"
            f"Примеры:\n"
            f"  /find vpn туннель          - поиск с threshold={default_threshold} (по умолчанию)\n"
            f"  /find 2.0 vpn туннель       - поиск с threshold=2.0\n"
            f"  /find 0.5 test              - поиск с threshold=0.5"
        )
    
    parts = text.split(maxsplit=1)
    threshold = default_threshold
    search_query = ""
    
    try:
        # Check if first argument is a number
        potential_threshold = float(parts[0].strip())
        threshold = potential_threshold
        # If threshold parsed successfully, query is the rest
        if len(parts) >= 2:
            search_query = parts[1].strip()
        else:
            return None, (
                "Использование: /find [thr] <запрос>\n\n"
                "Если указан threshold, необходимо также указать запрос.\n"
                "Пример: /find 2.0 vpn туннель"
            )
    except ValueError:
        # First argument is not a number, treat entire text as query
        search_query = text.strip()
    
    if not search_query:
        return None, (
            "Использование: /find [thr] <запрос>\n\n"
            "Необходимо указать запрос для поиска.\n"
            "Пример: /find vpn туннель"
        )
    
    return threshold, search_query


def handle_find_command_cli(args_text: str, bot, admin_manager, debug_rag: bool) -> str:
    """
    Handle /find command in CLI mode - output results to console.
    
    Args:
        args_text: Command arguments
        bot: LegaleBot instance
        admin_manager: AdminManager instance (can be None)
        debug_rag: Whether to show debug RAG info
        
    Returns:
        Response message
    """
    try:
        # Parse arguments
        threshold, query = parse_find_command_args(args_text, admin_manager)
        if threshold is None:
            return query  # query is error message
        
        # Simple search in chunk embeddings
        results = bot.retrieval.search_chunks_basic(query, n_results=100)
        
        # Filter results by cosine distance threshold
        filtered_results = [
            item for item in results 
            if float(item.get("distance", float('inf'))) <= threshold
        ]
        
        if not filtered_results:
            return f'по запросу "{query}" ничего не найдено (distance <= {threshold})'
        
        # Prepare message parts from filtered results
        from src.core.message_search import _prepare_message_parts
        message_parts_list = _prepare_message_parts(bot.db, filtered_results, debug_rag)
        
        if not message_parts_list:
            return f'по запросу "{query}" ничего не найдено'
        
        # Format and print results to console
        output_lines = [f'Найдено результатов по запросу "{query}" (threshold={threshold}):\n']
        output_lines.append("=" * 70)
        
        for idx, message_parts in enumerate(message_parts_list, 1):
            for part_idx, part in enumerate(message_parts, 1):
                # Remove HTML tags for console output
                content = part.get("content", "")
                # Simple HTML tag removal
                content = re.sub(r'<[^>]+>', '', content)
                
                output_lines.append(f"\n--- Результат {idx}, часть {part_idx} ---")
                output_lines.append(f"Distance: {part.get('distance', 'N/A')}")
                output_lines.append("-" * 70)
                output_lines.append(content)
                output_lines.append("-" * 70)
        
        syslog2(
            LOG_ALERT,
            "find command cli",
            query=query,
            threshold=threshold,
            chunks_found=len(filtered_results),
            messages=len(message_parts_list),
        )
        
        return "\n".join(output_lines)
        
    except Exception as e:
        syslog2(LOG_ERR, "find command cli failed", error=str(e))
        return f"Ошибка при выполнении поиска: {e}"


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
    
    # Count -v flags for log level (need to check before stream consumes them)
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

    syslog2(LOG_NOTICE, "cli bot initializing", model=model_name, log_level=syslog_level, chunks=chunks)
    
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
            log_level=syslog_level,
            debug_rag=debug_rag,
            profile_dir=profile_dir
        )
        
        # Create AdminManager if profile_dir is available (for config access)
        admin_manager = None
        if profile_dir:
            try:
                from src.bot.admin import AdminManager
                admin_manager = AdminManager(Path(profile_dir))
            except Exception as e:
                syslog2(LOG_WARNING, "admin manager init failed", error=str(e), action="continuing without admin_manager")
        
        print("Bot ready! Type 'exit' or 'quit' to stop.")
        print("Use /help to see available commands.")
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
            
            # Check if input is a command
            command_response = handle_cli_command(user_input, bot, admin_manager, debug_rag)
            if command_response is not None:
                # Command was handled
                print(f"Bot: {command_response}")
                print("-" * 50)
                continue
            
            # Not a command - handle as regular query
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
