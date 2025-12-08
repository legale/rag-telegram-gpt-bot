"""Tests for response formatter utilities."""
import pytest
from src.bot.utils.response_formatter import ResponseFormatter

def test_format_file_size():
    assert ResponseFormatter.format_file_size(0) == "0 B"
    assert ResponseFormatter.format_file_size(500) == "500 B"
    assert ResponseFormatter.format_file_size(1024) == "1.00 KB"
    assert ResponseFormatter.format_file_size(2048) == "2.00 KB"
    assert ResponseFormatter.format_file_size(1024*1024) == "1.00 MB"
    assert ResponseFormatter.format_file_size(5*1024*1024) == "5.00 MB"
    assert ResponseFormatter.format_file_size(1024*1024*1024) == "1.00 GB"

def test_format_percentage():
    assert ResponseFormatter.format_percentage(0, 100) == "0.0%"
    assert ResponseFormatter.format_percentage(50, 100) == "50.0%"
    assert ResponseFormatter.format_percentage(75, 100, decimals=2) == "75.00%"
    assert ResponseFormatter.format_percentage(0, 0) == "0.0%"

def test_create_progress_bar():
    bar = ResponseFormatter.create_progress_bar(0, 100)
    assert len(bar) == 20
    assert ResponseFormatter.create_progress_bar(50, 100).count("▓") == 10
    assert ResponseFormatter.create_progress_bar(0, 0).count("░") == 20

def test_format_error_message():
    assert "Ошибка:" in ResponseFormatter.format_error_message("test error")
    assert "контекст" in ResponseFormatter.format_error_message("error", "контекст")

def test_format_success_message():
    msg = ResponseFormatter.format_success_message("Success")
    assert msg == "Success"
    msg = ResponseFormatter.format_success_message("Success", {"key": "value"})
    assert "key: value" in msg

def test_format_info_warning_number():
    assert ResponseFormatter.format_info_message("info") == "info"
    assert ResponseFormatter.format_warning_message("warning") == "warning"
    assert ResponseFormatter.format_number(1234567) == "1,234,567"

