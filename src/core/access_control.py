"""
Access control logic for Legale Bot.
Handles admin authentication and access checks.
"""

from typing import Optional, Protocol
from pathlib import Path


class ConfigProvider(Protocol):
    """Protocol for configuration providers that supply admin password."""
    
    @property
    def admin_password(self) -> str:
        """Get admin password from configuration."""
        ...


class AdminAccessControl:
    """Handles admin access control logic."""
    
    def __init__(self, admin_store, config_provider: ConfigProvider):
        """
        Initialize AdminAccessControl.
        
        Args:
            admin_store: AdminStore instance for admin data
            config_provider: ConfigProvider instance for password access
        """
        self.admin_store = admin_store
        self.config_provider = config_provider
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin.
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            True if user is admin
        """
        if not self.admin_store.admin_exists():
            return False
        admin = self.admin_store.get_admin()
        return admin is not None and admin.get('user_id') == user_id
    
    def verify_password(self, password: str) -> bool:
        """
        Verify admin password.
        
        Args:
            password: Password to verify
        
        Returns:
            True if password is correct
        """
        return password == self.config_provider.admin_password
    
    def get_admin(self) -> Optional[dict]:
        """
        Get current admin info.
        
        Returns:
            Dict with admin info or None if no admin set
        """
        return self.admin_store.get_admin()

