import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.bot.cli import main

class TestCli:
    def test_main_exit(self):
        with patch('builtins.input', side_effect=['exit']), \
             patch('src.bot.cli.LegaleBot') as MockBot, \
             patch('os.getenv') as mock_getenv, \
             patch('argparse.ArgumentParser.parse_args') as mock_args:
             
             def get_env(key):
                 if key == "DATABASE_URL": return "sqlite:///db"
                 if key == "VECTOR_DB_PATH": return "/path/vec"
                 return "key"
             mock_getenv.side_effect = get_env
             
             mock_args.return_value.verbose = 0
             mock_args.return_value.chunks = 5
             
             main()
             MockBot.assert_called()

    def test_main_chat(self):
        with patch('builtins.input', side_effect=['Hello', 'exit']), \
             patch('src.bot.cli.LegaleBot') as MockBot, \
             patch('os.getenv') as mock_getenv, \
             patch('src.bot.cli.load_dotenv'), \
             patch('argparse.ArgumentParser.parse_args') as mock_args:
             
             def get_env(key):
                 if key == "DATABASE_URL": return "sqlite:///db"
                 if key == "VECTOR_DB_PATH": return "/path/vec"
                 return "key"
             mock_getenv.side_effect = get_env
             
             mock_args.return_value.verbose = 0
             mock_args.return_value.chunks = 5
             MockBot.return_value.chat.return_value = "Hi there"
             
             main()
             
             MockBot.return_value.chat.assert_called_with("Hello", n_results=5)

    def test_main_missing_env(self):
        with patch('os.getenv', return_value=None), \
             patch('src.bot.cli.load_dotenv'), \
             patch('builtins.print') as mock_print, \
             patch('argparse.ArgumentParser.parse_args') as mock_args:
             
             main()
             
             found_error = False
             for call in mock_print.call_args_list:
                 args, _ = call
                 if args and "DATABASE_URL and VECTOR_DB_PATH must be set" in str(args[0]):
                     found_error = True
                     break
             assert found_error
