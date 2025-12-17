
import unittest
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.db import Database, UserModel, MessageModel

class TestGetMessagesByUser(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_msgs_by_user_db.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.db = Database(self.db_url)
        
        # Setup data
        self.username = "target_user"
        self.aliases = ["Alias1", "Alias2"]
        self.db.add_user(self.username, self.aliases)
        
        self.base_time = datetime.now()
        
        # Add messages
        self.messages = [
            {"msg_id": "1", "chat_id": "chat1", "ts": self.base_time - timedelta(minutes=10), "from_id": "target_user", "text": "Msg 1"},
            {"msg_id": "2", "chat_id": "chat1", "ts": self.base_time - timedelta(minutes=9), "from_id": "Alias1", "text": "Msg 2"},
            {"msg_id": "3", "chat_id": "chat1", "ts": self.base_time - timedelta(minutes=8), "from_id": "OtherUser", "text": "Noise"},
            {"msg_id": "4", "chat_id": "chat1", "ts": self.base_time - timedelta(minutes=7), "from_id": "Alias2", "text": "Msg 3"},
            {"msg_id": "5", "chat_id": "chat2", "ts": self.base_time - timedelta(minutes=6), "from_id": "target_user", "text": "Msg 4"},
        ]
        
        for m in self.messages:
            self.db.add_message(m["msg_id"], m["chat_id"], m["ts"], m["from_id"], m["text"])

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_messages_user_and_aliases(self):
        msgs = self.db.get_messages_by_user(self.username)
        self.assertEqual(len(msgs), 4) # 1, 2, 4, 5
        
        # Check if sorted by ts
        self.assertEqual(msgs[0].msg_id, "1")
        self.assertEqual(msgs[1].msg_id, "2")
        self.assertEqual(msgs[2].msg_id, "4")
        self.assertEqual(msgs[3].msg_id, "5")

    def test_limit(self):
        msgs = self.db.get_messages_by_user(self.username, limit=2)
        self.assertEqual(len(msgs), 2)
        # Should return latest 2 (Msg 4, Msg 3) if logic is desc limit
        # But `get_messages_by_user` sorts chrono at the end.
        # Query: order_by DESC -> limit n -> all() -> sorted ASC
        # So it fetches N latest messages, then sorts them chronological.
        
        # Latest are "5" and "4".
        self.assertEqual(msgs[0].msg_id, "4")
        self.assertEqual(msgs[1].msg_id, "5")

    def test_date_filter(self):
        cutoff = self.base_time - timedelta(minutes=8, seconds=30)
        # Should exclude Msg 1 (-10m) and Msg 2 (-9m).
        # Should include Msg 4 (-7m) and Msg 5 (-6m). 
        # Wait, Msg 4 is "4" (Alias2) and Msg 5 is "5" (target_user).
        # Msg 1 is date -10m. Msg 2 is -9m.
        # Cutoff is -8m30s.
        
        msgs = self.db.get_messages_by_user(self.username, date_filter=cutoff)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].msg_id, "4")
        self.assertEqual(msgs[1].msg_id, "5")

if __name__ == "__main__":
    unittest.main()
