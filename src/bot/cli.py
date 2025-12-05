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

def main():
    # Load .env from project root
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    
    # Debug: Check if key is loaded
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print(f"Warning: Neither OPENROUTER_API_KEY nor OPENAI_API_KEY found in {dotenv_path}")
        # Print first few chars if exists to verify
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Legale Bot CLI")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (-v, -vv, -vvv)")
    parser.add_argument("--chunks", type=int, default=5, help="Number of context chunks to retrieve (default: 5)")
    args = parser.parse_args()

    # Read model from models.txt
    model_name = "openai/gpt-3.5-turbo" # Default
    try:
        with open("models.txt", "r") as f:
            line = f.readline().strip()
            if line:
                model_name = line
    except FileNotFoundError:
        print("Warning: models.txt not found, using default model.")

    print(f"Initializing Legale Bot with model: {model_name} (Verbosity: {args.verbose}, Chunks: {args.chunks})...")
    try:
        bot = LegaleBot(model_name=model_name, verbosity=args.verbose)
        print("Bot ready! Type 'exit' or 'quit' to stop.")
        print("-" * 50)
    except Exception as e:
        print(f"Error initializing bot: {e}")
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
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
