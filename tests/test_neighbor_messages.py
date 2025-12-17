
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

class TestNeighborMessages(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_neighbor_db.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_url = f"sqlite:///{self.db_path}"
        self.db = Database(self.db_url)
        
        self.base_time = datetime.now()
        self.chat_id = "test_chat"
        
        # Add target
        self.db.add_message("target", self.chat_id, self.base_time, "User", "Target Msg")
        
        # Add context (5 before, 5 after)
        for i in range(5): # 0,1,2,3,4. 0 is closest.
            # Before: -1m, -2m...
            ts = self.base_time - timedelta(minutes=i+1)
            self.db.add_message(f"before_{i}", self.chat_id, ts, "User", f"Before {i}")
            
            # After: +1m, +2m...
            ts = self.base_time + timedelta(minutes=i+1)
            self.db.add_message(f"after_{i}", self.chat_id, ts, "User", f"After {i}")
            
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_window_count(self):
        target = self.db.get_message_by_id("target")
        # Window count 2 -> should get 2 before, target, 2 after. Total 5.
        neighbors = self.db.get_neighbor_messages(target, window_count=2, max_tokens=1000)
        
        self.assertEqual(len(neighbors), 5)
        # Check order
        self.assertEqual(neighbors[0].msg_id, "before_1") # -2m
        self.assertEqual(neighbors[1].msg_id, "before_0") # -1m
        self.assertEqual(neighbors[2].msg_id, "target")
        self.assertEqual(neighbors[3].msg_id, "after_0") # +1m
        self.assertEqual(neighbors[4].msg_id, "after_1") # +2m

    def test_max_tokens_limit(self):
        target = self.db.get_message_by_id("target")
        # Target ~ 2 tokens ("Target Msg").
        # "Before 0" ~ 2 tokens ("Before 0"). Total 4 > 3. (Skipped)
        # "After 0" ~ 1 token ("After 0"). Total 3 <= 3. (Added)
        
        neighbors = self.db.get_neighbor_messages(target, window_count=5, max_tokens=3)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0].msg_id, "target")
        self.assertEqual(neighbors[1].msg_id, "after_0")

    def test_contiguity_stop(self):
        # Create a gap scenario
        # Target (2t)
        # Before 0 (Large, 10t) -> Should fail
        # Before 1 (Small, 1t) -> Should be skipped because Before 0 failed
        
        big_msg_ts = self.base_time - timedelta(minutes=10)
        self.db.add_message("big_msg", self.chat_id, big_msg_ts, "User", "A" * 40) # 10 tokens
        
        small_msg_ts = self.base_time - timedelta(minutes=11)
        self.db.add_message("small_msg", self.chat_id, small_msg_ts, "User", "A") # <1 token
        
        # We need them to be neighbors.
        # Overwrite setup's neighbors? Or just use a new chat.
        new_chat = "contiguity_chat"
        self.db.add_message("tgt", new_chat, self.base_time, "U", "Target") # 1-2 tokens
        target = self.db.get_message_by_id("tgt")
        
        # Before 0
        self.db.add_message("big", new_chat, self.base_time - timedelta(minutes=1), "U", "Big " * 10) # ~40 chars -> 10t
        # Before 1
        self.db.add_message("small", new_chat, self.base_time - timedelta(minutes=2), "U", "Small") # ~1t
        
        # Max tokens = 5. Target(~2) fits. Big(10) fails. Small(1) fits but shouldn't be reached.
        neighbors = self.db.get_neighbor_messages(target, window_count=5, max_tokens=5)
        
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].msg_id, "tgt")
        
    def test_day_boundary(self):
        # Create a message on previous day
        prev_day = self.base_time - timedelta(days=1, minutes=10)
        self.db.add_message("prev_day_msg", self.chat_id, prev_day, "User", "Old Msg")
        
        target = self.db.get_message_by_id("target")
        neighbors = self.db.get_neighbor_messages(target, window_count=10)
        
        ids = [m.msg_id for m in neighbors]
        self.assertNotIn("prev_day_msg", ids)

if __name__ == "__main__":
    unittest.main()
