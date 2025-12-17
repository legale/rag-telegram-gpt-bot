
import unittest
import os
import sys
import json
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.db import Database
from src.core.profiling import AliasDiscoveryService

class TestAliasDiscovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db_path = "test_alias_db.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.db = Database(self.db_url)
        
        self.bot_mock = MagicMock()
        self.bot_mock.complete = AsyncMock()
        
        self.service = AliasDiscoveryService(self.db, self.bot_mock)
        
        # Setup user and messages
        self.username = "target_agent"
        self.db.add_user(self.username)
        self.db.add_message("1", "chat1", datetime.now(), self.username, "Hello")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def test_discover_aliases_parsing(self):
        # Mock LLM response
        llm_response = '{"user": "target_agent", "alias": "\'Agent007\' \'James Bond\'"}'
        self.bot_mock.complete.return_value = llm_response
        
        aliases = await self.service.discover_aliases(self.username)
        
        self.assertEqual(len(aliases), 2)
        self.assertIn("Agent007", aliases)
        self.assertIn("James Bond", aliases)
        
        # Verify prompt construction (basic check)
        call_args = self.bot_mock.complete.call_args
        prompt = call_args[0][0]
        self.assertIn("Target User: target_agent", prompt)
        self.assertIn(AliasDiscoveryService.SYSTEM_PROMPT, call_args[1]['system_prompt'])

    async def test_discover_aliases_no_data(self):
        # User with no messages
        aliases = await self.service.discover_aliases("unknown_user")
        self.assertEqual(aliases, [])
        self.bot_mock.complete.assert_not_called()

    async def test_save_aliases(self):
        aliases = ["Bond", "007"]
        self.service.save_aliases(self.username, aliases)
        
        user = self.db.get_user(self.username)
        stored = json.loads(user.aliases)
        self.assertEqual(set(stored), set(aliases))
        
        # Test merge
        new_aliases = ["Spy"]
        self.service.save_aliases(self.username, new_aliases)
        
        user = self.db.get_user(self.username)
        stored = json.loads(user.aliases)
        self.assertEqual(len(stored), 3)
        self.assertIn("Spy", stored)

if __name__ == "__main__":
    unittest.main()
