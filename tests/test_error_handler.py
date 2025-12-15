"""
Tests for ErrorHandler utility.
"""

import pytest
from src.bot.utils.error_handler import ErrorHandler


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def test_handle_error_static(self):
        """Test static error handling method."""
        error = ValueError("Test error")
        result = ErrorHandler.handle_error_static(error, "test operation")
        
        assert "Ошибка" in result
        assert "test operation" in result
        assert "Test error" in result
    
    def test_handle_error_static_with_custom_message(self):
        """Test static error handling with custom user message."""
        error = ValueError("Test error")
        custom_message = "Custom error message"
        result = ErrorHandler.handle_error_static(
            error, "test operation", user_message=custom_message
        )
        
        assert result == custom_message
    
    def test_handle_error_instance(self):
        """Test instance error handling method."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        result = handler.handle_error(error, "test operation")
        
        assert "Ошибка" in result
        assert "test operation" in result
        assert "Test error" in result
    
    def test_handle_error_instance_with_custom_message(self):
        """Test instance error handling with custom user message."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        custom_message = "Custom error message"
        result = handler.handle_error(
            error, "test operation", user_message=custom_message
        )
        
        assert result == custom_message
    
    def test_handle_error_with_different_exceptions(self):
        """Test error handling with different exception types."""
        handler = ErrorHandler()
        
        errors = [
            ValueError("Value error"),
            KeyError("key"),
            AttributeError("attribute"),
            RuntimeError("Runtime error"),
        ]
        
        for error in errors:
            result = handler.handle_error(error, "test operation")
            assert "Ошибка" in result
            assert "test operation" in result

