"""
Error handling utility for Legale Bot.

Provides unified error handling across the application.
"""

from typing import Optional
from src.bot.utils.response_formatter import ResponseFormatter
from src.lib.syslog2 import *


class ErrorHandler:
    """Utility class for unified error handling."""
    
    def __init__(self):
        self.formatter = ResponseFormatter()
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        log_level: int = LOG_ERR,
        user_message: Optional[str] = None
    ) -> str:
        """
        Handle an error with unified logging and user message formatting.
        
        Args:
            error: Exception that occurred
            context: Context description (e.g., operation name)
            log_level: Logging level (default: LOG_ERR)
            user_message: Optional custom user message. If None, uses default formatting.
            
        Returns:
            Formatted error message for user
        """
        syslog2(log_level, "error occurred", context=context, error=str(error))
        
        if user_message:
            return user_message
        
        return self.formatter.format_error_message(str(error), context)
    
    @staticmethod
    def handle_error_static(
        error: Exception,
        context: str,
        log_level: int = LOG_ERR,
        user_message: Optional[str] = None
    ) -> str:
        """
        Static method version for convenience.
        
        Args:
            error: Exception that occurred
            context: Context description (e.g., operation name)
            log_level: Logging level (default: LOG_ERR)
            user_message: Optional custom user message. If None, uses default formatting.
            
        Returns:
            Formatted error message for user
        """
        formatter = ResponseFormatter()
        syslog2(log_level, "error occurred", context=context, error=str(error))
        
        if user_message:
            return user_message
        
        return formatter.format_error_message(str(error), context)

