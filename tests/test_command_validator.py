"""
Tests for command validator utilities.
"""

import pytest
from src.bot.utils.command_validator import CommandValidator


class TestValidateProfileName:
    """Tests for validate_profile_name."""
    
    def test_valid_profile_name(self):
        """Test valid profile names."""
        is_valid, error = CommandValidator.validate_profile_name("test_profile")
        assert is_valid is True
        assert error == ""
        
        is_valid, error = CommandValidator.validate_profile_name("my-profile")
        assert is_valid is True
        
        is_valid, error = CommandValidator.validate_profile_name("profile123")
        assert is_valid is True
    
    def test_empty_profile_name(self):
        """Test empty profile name."""
        is_valid, error = CommandValidator.validate_profile_name("")
        assert is_valid is False
        assert "не может быть пустым" in error
    
    def test_too_long_profile_name(self):
        """Test profile name exceeding length limit."""
        long_name = "a" * 51
        is_valid, error = CommandValidator.validate_profile_name(long_name)
        assert is_valid is False
        assert "слишком длинное" in error
    
    def test_invalid_characters(self):
        """Test profile name with invalid characters."""
        is_valid, error = CommandValidator.validate_profile_name("test@profile")
        assert is_valid is False
        assert "может содержать только" in error
        
        is_valid, error = CommandValidator.validate_profile_name("test profile")
        assert is_valid is False
    
    def test_reserved_names(self):
        """Test reserved profile names."""
        for reserved in ["default", "test", "temp", "tmp"]:
            is_valid, error = CommandValidator.validate_profile_name(reserved)
            assert is_valid is False
            assert "зарезервировано" in error
            
            # Case insensitive
            is_valid, error = CommandValidator.validate_profile_name(reserved.upper())
            assert is_valid is False


class TestValidateArgsCount:
    """Tests for validate_args_count."""
    
    def test_valid_count(self):
        """Test valid argument counts."""
        is_valid, error = CommandValidator.validate_args_count(["a", "b"], 2, 2)
        assert is_valid is True
        
        is_valid, error = CommandValidator.validate_args_count(["a"], 1, 3)
        assert is_valid is True
        
        is_valid, error = CommandValidator.validate_args_count(["a", "b", "c"], 1, None)
        assert is_valid is True
    
    def test_too_few_args(self):
        """Test insufficient arguments."""
        is_valid, error = CommandValidator.validate_args_count(["a"], 2, 3)
        assert is_valid is False
        assert "Недостаточно аргументов" in error
    
    def test_too_many_args(self):
        """Test too many arguments."""
        is_valid, error = CommandValidator.validate_args_count(["a", "b", "c"], 1, 2)
        assert is_valid is False
        assert "Слишком много аргументов" in error
    
    def test_usage_message(self):
        """Test usage message inclusion."""
        is_valid, error = CommandValidator.validate_args_count(["a"], 2, 2, "test <arg1> <arg2>")
        assert is_valid is False
        assert "Использование" in error


class TestValidateInteger:
    """Tests for validate_integer."""
    
    def test_valid_integer(self):
        """Test valid integer values."""
        is_valid, value, error = CommandValidator.validate_integer("123")
        assert is_valid is True
        assert value == 123
        
        is_valid, value, error = CommandValidator.validate_integer("0")
        assert is_valid is True
        assert value == 0
        
        is_valid, value, error = CommandValidator.validate_integer("-456")
        assert is_valid is True
        assert value == -456
    
    def test_invalid_integer(self):
        """Test invalid integer values."""
        is_valid, value, error = CommandValidator.validate_integer("abc")
        assert is_valid is False
        assert "целым числом" in error
        
        is_valid, value, error = CommandValidator.validate_integer("12.5")
        assert is_valid is False
    
    def test_min_value(self):
        """Test minimum value validation."""
        is_valid, value, error = CommandValidator.validate_integer("5", min_val=10)
        assert is_valid is False
        assert ">=" in error
        
        is_valid, value, error = CommandValidator.validate_integer("15", min_val=10)
        assert is_valid is True
    
    def test_max_value(self):
        """Test maximum value validation."""
        is_valid, value, error = CommandValidator.validate_integer("150", max_val=100)
        assert is_valid is False
        assert "<=" in error
        
        is_valid, value, error = CommandValidator.validate_integer("50", max_val=100)
        assert is_valid is True
    
    def test_custom_field_name(self):
        """Test custom field name in error message."""
        is_valid, value, error = CommandValidator.validate_integer("abc", field_name="Порт")
        assert is_valid is False
        assert "Порт" in error


class TestValidateChatId:
    """Tests for validate_chat_id."""
    
    def test_valid_chat_id(self):
        """Test valid chat IDs."""
        is_valid, value, error = CommandValidator.validate_chat_id("123456")
        assert is_valid is True
        assert value == 123456
        
        is_valid, value, error = CommandValidator.validate_chat_id("-987654")
        assert is_valid is True
        assert value == -987654
    
    def test_invalid_chat_id(self):
        """Test invalid chat ID format."""
        is_valid, value, error = CommandValidator.validate_chat_id("abc")
        assert is_valid is False
        assert "целым числом" in error
    
    def test_too_large_chat_id(self):
        """Test chat ID exceeding reasonable limit."""
        large_id = str(10**16)  # Too large
        is_valid, value, error = CommandValidator.validate_chat_id(large_id)
        assert is_valid is False
        assert "Недопустимый ID чата" in error


class TestValidateFrequency:
    """Tests for validate_frequency."""
    
    def test_valid_frequency(self):
        """Test valid frequency values."""
        is_valid, value, error = CommandValidator.validate_frequency("0")
        assert is_valid is True
        assert value == 0
        
        is_valid, value, error = CommandValidator.validate_frequency("10")
        assert is_valid is True
        assert value == 10
        
        is_valid, value, error = CommandValidator.validate_frequency("1000")
        assert is_valid is True
        assert value == 1000
    
    def test_invalid_frequency(self):
        """Test invalid frequency values."""
        is_valid, value, error = CommandValidator.validate_frequency("-1")
        assert is_valid is False
        assert ">=" in error
        
        is_valid, value, error = CommandValidator.validate_frequency("1001")
        assert is_valid is False
        assert "<=" in error


class TestValidateLogLines:
    """Tests for validate_log_lines."""
    
    def test_valid_log_lines(self):
        """Test valid log line counts."""
        is_valid, value, error = CommandValidator.validate_log_lines("1")
        assert is_valid is True
        assert value == 1
        
        is_valid, value, error = CommandValidator.validate_log_lines("100")
        assert is_valid is True
        assert value == 100
        
        is_valid, value, error = CommandValidator.validate_log_lines("200")
        assert is_valid is True
        assert value == 200
    
    def test_invalid_log_lines(self):
        """Test invalid log line counts."""
        is_valid, value, error = CommandValidator.validate_log_lines("0")
        assert is_valid is False
        
        is_valid, value, error = CommandValidator.validate_log_lines("201")
        assert is_valid is False

