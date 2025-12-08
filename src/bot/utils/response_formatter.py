"""
Response formatting utilities for Legale Bot.

Provides consistent formatting for bot responses including:
- File sizes
- Percentages
- Progress bars
- Error and success messages
"""

from typing import Optional, Dict, Any


class ResponseFormatter:
    """Utility class for formatting bot responses."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Convert bytes to human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string (e.g., "1.23 MB", "456 KB")
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    @staticmethod
    def format_percentage(value: float, total: float, decimals: int = 1) -> str:
        """
        Format a percentage value.
        
        Args:
            value: Current value
            total: Total value
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string (e.g., "75.5%")
        """
        if total == 0:
            return "0.0%"
        percentage = (value / total) * 100
        return f"{percentage:.{decimals}f}%"
    
    @staticmethod
    def create_progress_bar(current: int, total: int, width: int = 20, 
                          filled_char: str = "▓", empty_char: str = "░") -> str:
        """
        Create a text-based progress bar.
        
        Args:
            current: Current progress value
            total: Total value
            width: Width of the progress bar in characters
            filled_char: Character for filled portion
            empty_char: Character for empty portion
            
        Returns:
            Progress bar string
        """
        if total == 0:
            return empty_char * width
        
        filled_width = int((current / total) * width)
        empty_width = width - filled_width
        
        return filled_char * filled_width + empty_char * empty_width
    
    @staticmethod
    def format_error_message(error: str, context: str = "") -> str:
        """
        Format an error message consistently.
        
        Args:
            error: Error message
            context: Optional context (e.g., operation name)
            
        Returns:
            Formatted error message
        """
        if context:
            return f"Ошибка при {context}: {error}"
        return f"Ошибка: {error}"
    
    @staticmethod
    def format_success_message(message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """
        Format a success message consistently.
        
        Args:
            message: Success message
            details: Optional dictionary of details to include
            
        Returns:
            Formatted success message
        """
        result = f"{message}"
        
        if details:
            result += "\n\n"
            for key, value in details.items():
                result += f"{key}: {value}\n"
        
        return result
    
    @staticmethod
    def format_info_message(message: str) -> str:
        """
        Format an info message.
        
        Args:
            message: Info message
            
        Returns:
            Formatted info message
        """
        return f"{message}"
    
    @staticmethod
    def format_warning_message(message: str) -> str:
        """
        Format a warning message.
        
        Args:
            message: Warning message
            
        Returns:
            Formatted warning message
        """
        return f"{message}"
    
    @staticmethod
    def format_number(number: int) -> str:
        """
        Format a number with thousands separators.
        
        Args:
            number: Number to format
            
        Returns:
            Formatted number string (e.g., "1,234,567")
        """
        return f"{number:,}"
