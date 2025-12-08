"""
Tests for Telegram link generation utilities.
"""

import pytest
from src.bot.utils.telegram_links import build_message_link


def test_build_message_link_with_username():
    """Test building link with username."""
    link = build_message_link(123456, 100, "test_channel")
    
    assert link == "https://t.me/test_channel/100"


def test_build_message_link_with_username_positive_chat_id():
    """Test that username takes precedence even with positive chat_id."""
    link = build_message_link(123456, 200, "my_channel")
    
    assert link == "https://t.me/my_channel/200"


def test_build_message_link_negative_chat_id():
    """Test building link with negative chat_id (group/channel)."""
    # Negative without 100 prefix returns tg:// URL
    link = build_message_link(-1234567890, 50, None)
    assert "tg://openmessage" in link
    assert "chat_id=-1234567890" in link


def test_build_message_link_negative_with_100_prefix():
    """Test handling negative chat_id with '100' prefix."""
    # Telegram format: -1001234567890 should become t.me/c/1234567890
    link = build_message_link(-1001234567890, 100, None)
    assert link == "https://t.me/c/1234567890/100"


def test_build_message_link_negative_small_number():
    """Test negative chat_id without '100' prefix returns tg:// URL."""
    link = build_message_link(-987654, 200, None)
    assert "tg://openmessage" in link
    assert "chat_id=-987654" in link


def test_build_message_link_positive_chat_id():
    """Test building link with positive chat_id returns tg:// URL."""
    link = build_message_link(1234567890, 300, None)
    assert "tg://openmessage" in link
    assert "user_id=1234567890" in link


def test_build_message_link_empty_username():
    """Test that empty username string is treated as None."""
    link = build_message_link(123456, 100, "")
    # Empty username should fall back to tg:// format for positive chat_id
    assert "tg://openmessage" in link


def test_build_message_link_whitespace_username():
    """Test that whitespace-only username is treated as None."""
    link = build_message_link(123456, 100, "   ")
    # Whitespace username should fall back to tg:// format
    assert "tg://openmessage" in link


def test_build_message_link_zero_chat_id():
    """Test edge case with zero chat_id."""
    link = build_message_link(0, 100, None)
    # Zero chat_id (positive) returns tg:// URL
    assert "tg://openmessage" in link
    assert "user_id=0" in link


def test_build_message_link_large_negative_chat_id():
    """Test with very large negative chat_id."""
    link = build_message_link(-999999999999999, 500, None)
    # Large negative without 100 prefix returns tg:// URL
    assert "tg://openmessage" in link
    assert "chat_id=-999999999999999" in link

