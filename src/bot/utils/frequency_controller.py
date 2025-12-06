"""
Frequency control utilities for Legale Bot.

Manages response frequency logic including:
- Message counting per chat
- Frequency-based response decisions
- Mention detection
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FrequencyController:
    """Controller for managing bot response frequency."""
    
    def __init__(self):
        """Initialize FrequencyController."""
        self.chat_counters: Dict[int, int] = {}
    
    def should_respond(self, chat_id: int, frequency: int, 
                      has_mention: bool, is_command: bool, 
                      is_private: bool) -> tuple[bool, str]:
        """
        Determine if bot should respond based on frequency settings.
        
        Logic:
        - Commands: always respond (handled separately)
        - Private chats: always respond (if access granted)
        - Mentions: always respond
        - Frequency < 1: only respond to mentions
        - Frequency == 1: respond to every message
        - Frequency > 1: respond every Nth message
        
        Args:
            chat_id: Telegram chat ID
            frequency: Response frequency setting (0 = only mentions, 1 = all, N = every Nth)
            has_mention: True if bot was mentioned
            is_command: True if message is a command
            is_private: True if private chat
            
        Returns:
            Tuple of (should_respond, reason)
        """
        # Commands and private chats are handled separately
        if is_command or is_private:
            return True, "command_or_private"
        
        # Bot mentioned: always respond
        if has_mention:
            logger.debug(f"Responding: bot mentioned in chat {chat_id}")
            return True, "mentioned"
        
        # Frequency < 1: only respond to mentions
        if frequency < 1:
            logger.debug(f"Skipping: freq<1 and no mention in chat {chat_id}")
            return False, "freq_zero_no_mention"
        
        # Frequency == 1: respond to all messages
        if frequency == 1:
            logger.debug(f"Responding: freq=1 in chat {chat_id}")
            return True, "freq_one"
        
        # Frequency > 1: respond every Nth message
        current = self.chat_counters.get(chat_id, 0) + 1
        self.chat_counters[chat_id] = current
        
        if current % frequency == 0:
            logger.debug(f"Responding: frequency match (msg {current}, freq {frequency}, chat {chat_id})")
            return True, f"freq_match_{current}"
        else:
            logger.debug(f"Skipping: frequency mismatch (msg {current}, freq {frequency}, chat {chat_id})")
            return False, f"freq_skip_{current}"
    
    def reset_counter(self, chat_id: int):
        """
        Reset message counter for a chat.
        
        Args:
            chat_id: Telegram chat ID
        """
        if chat_id in self.chat_counters:
            del self.chat_counters[chat_id]
            logger.debug(f"Reset counter for chat {chat_id}")
    
    def get_counter(self, chat_id: int) -> int:
        """
        Get current message counter for a chat.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Current counter value
        """
        return self.chat_counters.get(chat_id, 0)
