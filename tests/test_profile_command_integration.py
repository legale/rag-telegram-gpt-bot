
import sys
import os
import unittest
from unittest.mock import Mock, AsyncMock
from types import SimpleNamespace

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.commands.user import ProfileCommandHandler
from src.core.dispatcher import CommandContext
import asyncio
from datetime import datetime

class TestProfileCommandIntegration(unittest.TestCase):
    def setUp(self):
        self.bot = SimpleNamespace(db=Mock(), complete=AsyncMock(), log_level=6) # 6=INFO
        self.command_handler = ProfileCommandHandler(self.bot)

    def test_profile_flow_success(self):
        # 1. Setup User and Messages
        user_mock = SimpleNamespace(username="testuser")
        self.bot.db.get_user.return_value = user_mock

        msg1 = SimpleNamespace(
            msg_id=1,
            ts=datetime(2023, 1, 1, 10, 0),
            from_id="testuser",
            text="Hello world",
        )
        
        self.bot.db.get_messages_by_user.return_value = [msg1]
        
        # Neighbor messages (context)
        neighbor1 = SimpleNamespace(
            msg_id=1,
            ts=msg1.ts,
            from_id="testuser",
            text="Hello world",
        )

        neighbor_ctx = SimpleNamespace(
            msg_id=2,
            ts=msg1.ts,
            from_id="other",
            text="Hi there",
        )
        
        self.bot.db.get_neighbor_messages.return_value = [neighbor1, neighbor_ctx]
        
        # Mock LLM response
        expected_profile = "User Profile: Analytical Thinker..."
        self.bot.complete.return_value = expected_profile
        
        # 2. Execute Command
        context = CommandContext(command_name="/userprofile", args=["testuser"])
        
        # Run async handle
        result = asyncio.run(self.command_handler.handle(context))
        
        # 3. Verify
        self.assertTrue(result.success)
        self.assertEqual(result.message, expected_profile)
        
        # Verify calls
        self.bot.db.get_user.assert_called_with("testuser")
        self.bot.db.get_messages_by_user.assert_called_with("testuser", limit=30)
        self.bot.db.get_neighbor_messages.assert_called()
        
        # Verify LLM prompt contains context
        call_args = self.bot.complete.call_args
        prompt = call_args[0][0]
        self.assertIn("Целевой пользователь: testuser", prompt)
        self.assertIn("[user: testuser] Hello world", prompt)
        self.assertIn("[user: other] Hi there", prompt)

    def test_profile_user_not_found(self):
        self.bot.db.get_user.return_value = None
        self.bot.db.get_user_by_alias.return_value = None
        
        context = CommandContext(command_name="/userprofile", args=["unknown"])
        result = asyncio.run(self.command_handler.handle(context))
        
        self.assertFalse(result.success)
        self.assertIn("не найден", result.message)

    def test_profile_no_messages(self):
        user_mock = SimpleNamespace(username="silentuser")
        self.bot.db.get_user.return_value = user_mock
        self.bot.db.get_messages_by_user.return_value = []
        
        context = CommandContext(command_name="/userprofile", args=["silentuser"])
        result = asyncio.run(self.command_handler.handle(context))
        
        self.assertFalse(result.success)
        self.assertIn("Нет сообщений", result.message)

    def test_profile_joins_multiword_alias(self):
        user_mock = SimpleNamespace(username="testuser")
        self.bot.db.get_user.return_value = None
        self.bot.db.get_user_by_alias.return_value = user_mock
        self.bot.db.get_messages_by_user.return_value = []

        context = CommandContext(command_name="/userprofile", args=["James", "Bond"])
        result = asyncio.run(self.command_handler.handle(context))

        self.bot.db.get_user.assert_called_with("James Bond")
        self.bot.db.get_user_by_alias.assert_called_with("James Bond")
        self.assertFalse(result.success)
        self.assertIn("Нет сообщений", result.message)

if __name__ == '__main__':
    unittest.main()
