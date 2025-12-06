"""
Tests for FrequencyController.

Tests cover:
- Frequency-based response decisions
- Mention detection logic
- Counter increment and tracking
- Private vs group chat behavior
- Command handling
"""

import pytest
from src.bot.utils.frequency_controller import FrequencyController


class TestShouldRespond:
    """Tests for should_respond() method."""
    
    def test_command_always_responds(self):
        """Test commands always trigger response."""
        controller = FrequencyController()
        should_respond, reason = controller.should_respond(
            chat_id=-100123456,
            frequency=0,  # Even with freq=0
            has_mention=False,
            is_command=True,
            is_private=False
        )
        
        assert should_respond is True
        assert reason == "command_or_private"
    
    def test_private_always_responds(self):
        """Test private chats always trigger response."""
        controller = FrequencyController()
        should_respond, reason = controller.should_respond(
            chat_id=12345,
            frequency=0,  # Even with freq=0
            has_mention=False,
            is_command=False,
            is_private=True
        )
        
        assert should_respond is True
        assert reason == "command_or_private"
    
    def test_mention_always_responds(self):
        """Test mentions always trigger response."""
        controller = FrequencyController()
        should_respond, reason = controller.should_respond(
            chat_id=-100123456,
            frequency=0,  # Even with freq=0
            has_mention=True,
            is_command=False,
            is_private=False
        )
        
        assert should_respond is True
        assert reason == "mentioned"
    
    def test_frequency_zero_no_mention_skips(self):
        """Test freq=0 without mention skips response."""
        controller = FrequencyController()
        should_respond, reason = controller.should_respond(
            chat_id=-100123456,
            frequency=0,
            has_mention=False,
            is_command=False,
            is_private=False
        )
        
        assert should_respond is False
        assert reason == "freq_zero_no_mention"
    
    def test_frequency_negative_no_mention_skips(self):
        """Test negative frequency without mention skips response."""
        controller = FrequencyController()
        should_respond, reason = controller.should_respond(
            chat_id=-100123456,
            frequency=-1,
            has_mention=False,
            is_command=False,
            is_private=False
        )
        
        assert should_respond is False
        assert reason == "freq_zero_no_mention"
    
    def test_frequency_one_always_responds(self):
        """Test freq=1 always responds to all messages."""
        controller = FrequencyController()
        
        # Multiple messages should all get responses
        for i in range(5):
            should_respond, reason = controller.should_respond(
                chat_id=-100123456,
                frequency=1,
                has_mention=False,
                is_command=False,
                is_private=False
            )
            
            assert should_respond is True
            assert reason == "freq_one"
    
    def test_frequency_two_every_second_message(self):
        """Test freq=2 responds every 2nd message."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Message 1: skip
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        assert "freq_skip_1" in reason
        
        # Message 2: respond
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True
        assert "freq_match_2" in reason
        
        # Message 3: skip
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        assert "freq_skip_3" in reason
        
        # Message 4: respond
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True
        assert "freq_match_4" in reason
    
    def test_frequency_five_every_fifth_message(self):
        """Test freq=5 responds every 5th message."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Messages 1-4: skip
        for i in range(1, 5):
            should_respond, reason = controller.should_respond(
                chat_id=chat_id, frequency=5, has_mention=False,
                is_command=False, is_private=False
            )
            assert should_respond is False
            assert f"freq_skip_{i}" in reason
        
        # Message 5: respond
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=5, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True
        assert "freq_match_5" in reason
        
        # Messages 6-9: skip
        for i in range(6, 10):
            should_respond, reason = controller.should_respond(
                chat_id=chat_id, frequency=5, has_mention=False,
                is_command=False, is_private=False
            )
            assert should_respond is False
            assert f"freq_skip_{i}" in reason
        
        # Message 10: respond
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=5, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True
        assert "freq_match_10" in reason
    
    def test_frequency_separate_chat_counters(self):
        """Test counters are separate for different chats."""
        controller = FrequencyController()
        chat1 = -100111111
        chat2 = -100222222
        
        # Chat 1: message 1 (skip)
        should_respond, _ = controller.should_respond(
            chat_id=chat1, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        
        # Chat 2: message 1 (skip)
        should_respond, _ = controller.should_respond(
            chat_id=chat2, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        
        # Chat 1: message 2 (respond)
        should_respond, _ = controller.should_respond(
            chat_id=chat1, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True
        
        # Chat 2: message 2 (respond)
        should_respond, _ = controller.should_respond(
            chat_id=chat2, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is True


class TestResetCounter:
    """Tests for reset_counter() method."""
    
    def test_reset_counter_existing_chat(self):
        """Test resetting counter for existing chat."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Increment counter
        controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert controller.get_counter(chat_id) == 1
        
        # Reset
        controller.reset_counter(chat_id)
        assert controller.get_counter(chat_id) == 0
    
    def test_reset_counter_nonexistent_chat(self):
        """Test resetting counter for chat that doesn't exist."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Reset non-existent chat (should not raise error)
        controller.reset_counter(chat_id)
        assert controller.get_counter(chat_id) == 0
    
    def test_reset_counter_restarts_frequency(self):
        """Test counter reset restarts frequency counting."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Message 1: skip
        should_respond, _ = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        
        # Reset counter
        controller.reset_counter(chat_id)
        
        # Message 1 again: skip (counter restarted)
        should_respond, reason = controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=False
        )
        assert should_respond is False
        assert "freq_skip_1" in reason


class TestGetCounter:
    """Tests for get_counter() method."""
    
    def test_get_counter_initial_zero(self):
        """Test counter is initially 0 for new chat."""
        controller = FrequencyController()
        assert controller.get_counter(-100123456) == 0
    
    def test_get_counter_after_messages(self):
        """Test counter increments after messages."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Send 3 messages
        for i in range(3):
            controller.should_respond(
                chat_id=chat_id, frequency=5, has_mention=False,
                is_command=False, is_private=False
            )
        
        assert controller.get_counter(chat_id) == 3
    
    def test_get_counter_multiple_chats(self):
        """Test counters are tracked separately for multiple chats."""
        controller = FrequencyController()
        chat1 = -100111111
        chat2 = -100222222
        
        # Chat 1: 2 messages
        for i in range(2):
            controller.should_respond(
                chat_id=chat1, frequency=5, has_mention=False,
                is_command=False, is_private=False
            )
        
        # Chat 2: 5 messages
        for i in range(5):
            controller.should_respond(
                chat_id=chat2, frequency=5, has_mention=False,
                is_command=False, is_private=False
            )
        
        assert controller.get_counter(chat1) == 2
        assert controller.get_counter(chat2) == 5
    
    def test_get_counter_not_incremented_by_commands(self):
        """Test counter not incremented for commands."""
        controller = FrequencyController()
        chat_id = -100123456
        
        # Send command (should not increment counter)
        controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=True, is_private=False
        )
        
        assert controller.get_counter(chat_id) == 0
    
    def test_get_counter_not_incremented_by_private(self):
        """Test counter not incremented for private chats."""
        controller = FrequencyController()
        chat_id = 12345
        
        # Send private message (should not increment counter)
        controller.should_respond(
            chat_id=chat_id, frequency=2, has_mention=False,
            is_command=False, is_private=True
        )
        
        assert controller.get_counter(chat_id) == 0
