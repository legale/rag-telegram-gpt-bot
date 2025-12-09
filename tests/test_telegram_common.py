"""
Tests for Telegram common utilities.
"""

import pytest
from src.bot.utils.telegram_common import (
    MAX_TG_CONTENT_LEN,
    format_message_html,
    split_message_if_needed
)


def test_max_tg_content_len():
    """Test MAX_TG_CONTENT_LEN constant."""
    assert MAX_TG_CONTENT_LEN == 4096


def test_format_message_html_simple():
    """Test HTML formatting of simple message."""
    msg_data = {
        "text": "Hello world",
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "Ru",
        "sender_id": None
    }
    
    result = format_message_html(msg_data, 571052)
    
    assert "id: 571052" in result
    assert "date: 2025-12-09T00:49:22+00:00" in result
    assert "sender: Ru" in result
    assert "<pre>Hello world</pre>" in result
    assert "<code>" in result


def test_format_message_html_with_sender_id():
    """Test HTML formatting with sender_id."""
    msg_data = {
        "text": "Test message",
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "User",
        "sender_id": 123456
    }
    
    result = format_message_html(msg_data, 100)
    
    assert "user_id: 123456" in result
    assert "sender: User" in result


def test_format_message_html_escapes_html():
    """Test that HTML characters are escaped."""
    msg_data = {
        "text": "<script>alert('xss')</script>",
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "User",
        "sender_id": None
    }
    
    result = format_message_html(msg_data, 1)
    
    assert "<script>" not in result
    assert "&lt;script&gt;" in result or "&lt;" in result


def test_split_message_if_needed_short_message():
    """Test that short messages are not split."""
    msg_data = {
        "text": "Short message",
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "Ru",
        "sender_id": None
    }
    
    result = split_message_if_needed(msg_data, 571052, MAX_TG_CONTENT_LEN)
    
    assert len(result) == 1
    assert result[0]["id"] == 571052
    assert result[0]["date"] == "2025-12-09T00:49:22+00:00"
    assert result[0]["sender"] == "Ru"
    assert "part" not in result[0]
    assert "Short message" in result[0]["content"]


def test_split_message_if_needed_long_message():
    """Test that long messages are split into parts."""
    # Create a very long message
    long_text = "A" * (MAX_TG_CONTENT_LEN + 1000)
    msg_data = {
        "text": long_text,
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "Ru",
        "sender_id": None
    }
    
    result = split_message_if_needed(msg_data, 571052, MAX_TG_CONTENT_LEN)
    
    # Should be split into multiple parts
    assert len(result) > 1
    
    # Check first part
    assert result[0]["id"] == 571052
    assert result[0]["part"] == 1
    assert "part: 1" in result[0]["content"]
    
    # Check last part
    assert result[-1]["part"] == len(result)
    assert "part: {}".format(len(result)) in result[-1]["content"]
    
    # All parts should have same id, date, sender
    for part in result:
        assert part["id"] == 571052
        assert part["date"] == "2025-12-09T00:49:22+00:00"
        assert part["sender"] == "Ru"


def test_split_message_if_needed_custom_max_len():
    """Test splitting with custom max length."""
    msg_data = {
        "text": "A" * 2000,
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "Ru",
        "sender_id": None
    }
    
    result = split_message_if_needed(msg_data, 571052, max_len=500)
    
    # Should be split into multiple parts
    assert len(result) > 1
    
    # Each part should be within max_len (approximately)
    for part in result:
        assert len(part["content"]) <= 600  # Allow some overhead for HTML tags


def test_split_message_if_needed_empty_text():
    """Test splitting message with empty text."""
    msg_data = {
        "text": "",
        "date": "2025-12-09T00:49:22+00:00",
        "sender": "Ru",
        "sender_id": None
    }
    
    result = split_message_if_needed(msg_data, 571052, MAX_TG_CONTENT_LEN)
    
    assert len(result) == 1
    assert result[0]["id"] == 571052
    assert "<pre></pre>" in result[0]["content"]

