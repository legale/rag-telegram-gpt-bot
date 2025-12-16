"""
Frequency control utilities for Legale Bot.

Manages response frequency logic including:
- Message counting per chat
- Frequency-based response decisions
- Mention detection
"""

import logging
from typing import Dict
from src.lib.syslog2 import *
from src.core.rate_limit import allow as core_allow


class FrequencyController:
    """Controller for managing bot response frequency."""
    
    def __init__(self):
        """Initialize FrequencyController."""
        self.chat_counters: Dict[int, int] = {}
    
    def _check_frequency_limit(self, chat_id: int, frequency: int) -> tuple[bool, str]:
        return core_allow(
            chat_id=chat_id,
            frequency=frequency,
            has_mention=False,
            is_command=False,
            is_private=False,
            chat_counters=self.chat_counters,
        )
    
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
        should, reason = core_allow(
            chat_id=chat_id,
            frequency=frequency,
            has_mention=has_mention,
            is_command=is_command,
            is_private=is_private,
            chat_counters=self.chat_counters,
        )
        if should and reason == "mentioned":
            syslog2(LOG_DEBUG, "responding mentioned", chat_id=chat_id)
        if should and reason == "freq_one":
            syslog2(LOG_DEBUG, "responding freq one", chat_id=chat_id)
        if reason.startswith("freq_match_"):
            syslog2(LOG_DEBUG, "responding freq match", chat_id=chat_id, reason=reason, freq=frequency)
        if reason.startswith("freq_skip_"):
            syslog2(LOG_DEBUG, "skipping freq mismatch", chat_id=chat_id, reason=reason, freq=frequency)
        if reason == "freq_zero_no_mention":
            syslog2(LOG_DEBUG, "skipping freq zero", chat_id=chat_id)
        return should, reason
    
    def reset_counter(self, chat_id: int):
        """
        Reset message counter for a chat.
        
        Args:
            chat_id: Telegram chat ID
        """
        if chat_id in self.chat_counters:
            del self.chat_counters[chat_id]
            syslog2(LOG_DEBUG, "reset counter", chat_id=chat_id)
    
    def get_counter(self, chat_id: int) -> int:
        """
        Get current message counter for a chat.
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            Current counter value
        """
        return self.chat_counters.get(chat_id, 0)
