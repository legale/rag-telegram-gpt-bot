
import unittest
import os
import sys
import json
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.db import Database, UserModel

class TestUsersTable(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_users_db.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.db = Database(self.db_url)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_get_user(self):
        username = "test_user_1"
        self.db.add_user(username)
        
        user = self.db.get_user(username)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, username)
        self.assertIsNone(user.aliases)

    def test_add_user_with_aliases(self):
        username = "test_user_2"
        aliases = ["alias1", "alias2"]
        self.db.add_user(username, aliases)
        
        user = self.db.get_user(username)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, username)
        
        stored_aliases = json.loads(user.aliases)
        self.assertEqual(stored_aliases, aliases)

    def test_update_user_aliases(self):
        username = "test_user_3"
        self.db.add_user(username)
        
        aliases = ["new_alias"]
        self.db.update_user_aliases(username, aliases)
        
        user = self.db.get_user(username)
        stored_aliases = json.loads(user.aliases)
        self.assertEqual(stored_aliases, aliases)

    def test_get_all_users(self):
        self.db.add_user("user_a")
        self.db.add_user("user_b")
        
        users = self.db.get_all_users()
        self.assertEqual(len(users), 2)
        usernames = sorted([u.username for u in users])
        self.assertEqual(usernames, ["user_a", "user_b"])

    def test_duplicate_user(self):
        # Adding same user twice should effectively check/update, not fail unique constraint (handled in add_user)
        self.db.add_user("user_x")
        self.db.add_user("user_x", ["alias_x"])
        
        users = self.db.get_all_users()
        self.assertEqual(len(users), 1)
        
        user = self.db.get_user("user_x")
        self.assertEqual(json.loads(user.aliases), ["alias_x"])

if __name__ == "__main__":
    unittest.main()
