"""
Command validation utilities for Legale Bot.

Provides centralized validation for command arguments including:
- Profile names
- Argument counts
- Integer values
- Chat IDs
"""

import re
from typing import Tuple, List, Optional


class CommandValidator:
    """Utility class for validating command arguments."""
    
    # Valid profile name pattern: alphanumeric, underscores, hyphens
    PROFILE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    @staticmethod
    def validate_profile_name(name: str) -> Tuple[bool, str]:
        """
        Validate a profile name.
        
        Args:
            name: Profile name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Имя профиля не может быть пустым"
        
        if len(name) > 50:
            return False, "Имя профиля слишком длинное (макс. 50 символов)"
        
        if not CommandValidator.PROFILE_NAME_PATTERN.match(name):
            return False, "Имя профиля может содержать только буквы, цифры, '_' и '-'"
        
        # Reserved names
        reserved = ["default", "test", "temp", "tmp"]
        if name.lower() in reserved:
            return False, f"Имя '{name}' зарезервировано"
        
        return True, ""
    
    @staticmethod
    def validate_args_count(args: List[str], min_count: int, 
                          max_count: Optional[int] = None,
                          usage: str = "") -> Tuple[bool, str]:
        """
        Validate the number of command arguments.
        
        Args:
            args: List of arguments
            min_count: Minimum required arguments
            max_count: Maximum allowed arguments (None = unlimited)
            usage: Usage string to show in error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        arg_count = len(args)
        
        if arg_count < min_count:
            error = f"Недостаточно аргументов (требуется минимум {min_count})"
            if usage:
                error += f"\n\nИспользование: {usage}"
            return False, error
        
        if max_count is not None and arg_count > max_count:
            error = f"Слишком много аргументов (максимум {max_count})"
            if usage:
                error += f"\n\nИспользование: {usage}"
            return False, error
        
        return True, ""
    
    @staticmethod
    def validate_integer(value: str, min_val: Optional[int] = None, 
                        max_val: Optional[int] = None,
                        field_name: str = "Значение") -> Tuple[bool, int, str]:
        """
        Validate and parse an integer value.
        
        Args:
            value: String value to parse
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field for error messages
            
        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        try:
            parsed = int(value)
        except ValueError:
            return False, 0, f"{field_name} должно быть целым числом"
        
        if min_val is not None and parsed < min_val:
            return False, 0, f"{field_name} должно быть >= {min_val}"
        
        if max_val is not None and parsed > max_val:
            return False, 0, f"{field_name} должно быть <= {max_val}"
        
        return True, parsed, ""
    
    @staticmethod
    def validate_chat_id(value: str) -> Tuple[bool, int, str]:
        """
        Validate and parse a Telegram chat ID.
        
        Args:
            value: String value to parse
            
        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        try:
            chat_id = int(value)
        except ValueError:
            return False, 0, "ID чата должен быть целым числом"
        
        # Telegram chat IDs can be negative (groups) or positive (users/channels)
        # Reasonable range check
        if abs(chat_id) > 10**15:
            return False, 0, "Недопустимый ID чата"
        
        return True, chat_id, ""
    
    @staticmethod
    def validate_frequency(value: str) -> Tuple[bool, int, str]:
        """
        Validate response frequency value.
        
        Args:
            value: String value to parse
            
        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        is_valid, parsed, error = CommandValidator.validate_integer(
            value, 
            min_val=0, 
            max_val=1000,
            field_name="Частота"
        )
        
        if not is_valid:
            return False, 0, error
        
        return True, parsed, ""
    
    @staticmethod
    def validate_log_lines(value: str) -> Tuple[bool, int, str]:
        """
        Validate number of log lines to display.
        
        Args:
            value: String value to parse
            
        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        is_valid, parsed, error = CommandValidator.validate_integer(
            value,
            min_val=1,
            max_val=200,
            field_name="Количество строк"
        )
        
        if not is_valid:
            return False, 0, error
        
        return True, parsed, ""
