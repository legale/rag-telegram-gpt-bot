
import sys
import os
import unittest
from unittest.mock import MagicMock, AsyncMock

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
        self.bot = MagicMock()
        self.bot.complete = AsyncMock()
        self.command_handler = ProfileCommandHandler(self.bot)

    def test_profile_flow_success(self):
        # 1. Setup User and Messages
        user_mock = MagicMock()
        user_mock.username = "testuser"
        self.bot.db.get_user.return_value = user_mock
        
        msg1 = MagicMock()
        msg1.msg_id = 1
        msg1.ts = datetime(2023, 1, 1, 10, 0)
        msg1.from_id = "testuser"
        msg1.text = "Hello world"
        
        self.bot.db.get_messages_by_user.return_value = [msg1]
        
        # Neighbor messages (context)
        neighbor1 = MagicMock()
        neighbor1.msg_id = 1
        neighbor1.ts = msg1.ts
        neighbor1.from_id = "testuser"
        neighbor1.text = "Hello world"
        
        neighbor_ctx = MagicMock()
        neighbor_ctx.msg_id = 2
        neighbor_ctx.ts = msg1.ts
        neighbor_ctx.from_id = "other"
        neighbor_ctx.text = "Hi there"
        
        self.bot.db.get_neighbor_messages.return_value = [neighbor1, neighbor_ctx]
        
        # Mock LLM response
        expected_profile = "User Profile: Analytical Thinker..."
        self.bot.complete.return_value = expected_profile
        
        # 2. Execute Command
        context = CommandContext(command_name="/profile", args=["testuser"])
        
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
        self.assertIn("Target User: testuser", prompt)
        self.assertIn("[user: testuser] Hello world", prompt)
        self.assertIn("[user: other] Hi there", prompt)

    def test_profile_user_not_found(self):
        self.bot.db.get_user.return_value = None
        self.bot.db.get_user_by_alias.return_value = None
        
        context = CommandContext(command_name="/profile", args=["unknown"])
        result = asyncio.run(self.command_handler.handle(context))
        
        self.assertFalse(result.success)
        self.assertIn("не найден", result.message)

    def test_profile_no_messages(self):
        user_mock = MagicMock()
        user_mock.username = "silentuser"
        self.bot.db.get_user.return_value = user_mock
        self.bot.db.get_messages_by_user.return_value = []
        
        context = CommandContext(command_name="/profile", args=["silentuser"])
        result = asyncio.run(self.command_handler.handle(context))
        
        self.assertFalse(result.success)
        self.assertIn("Нет сообщений", result.message)

if __name__ == '__main__':
    unittest.main()
