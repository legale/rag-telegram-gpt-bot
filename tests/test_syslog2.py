"""Tests for syslog2 logging utilities."""
import pytest
from unittest.mock import patch
from src.core.syslog2 import *

def test_setup_log():
    setup_log(LOG_DEBUG)
    setup_log(LOG_INFO)

def test_syslog2_filtering(caplog):
    setup_log(LOG_WARNING)
    syslog2(LOG_DEBUG, "debug message")  # Should be filtered
    syslog2(LOG_WARNING, "warning message")  # Should pass
    assert "warning message" in caplog.text

def test_format_params():
    assert _format_params({}) == ""
    result = _format_params({"key": "value"})
    assert "key" in result and "value" in result

def test_get_caller_info():
    file, line, func = _get_caller_info()
    assert isinstance(file, str)
    assert isinstance(line, int)

def test_syslog2_levels():
    setup_log(LOG_DEBUG)
    syslog2(LOG_ERR, "error")
    syslog2(LOG_WARNING, "warning")
    syslog2(LOG_INFO, "info")

