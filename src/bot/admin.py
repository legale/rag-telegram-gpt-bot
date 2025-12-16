"""
Admin management utilities for Legale Bot.
Handles admin authentication and storage.

DEPRECATED: This class is kept for backward compatibility.
New code should use AdminStore (src/app/config_store.py) and AdminAccessControl (src/core/access_control.py).
"""

import os
from pathlib import Path
from typing import Optional, Dict

from src.bot.config import BotConfig
from src.app.config_store import AdminStore
from src.core.access_control import AdminAccessControl

class AdminManager:
    """
    Manages bot administrators and configuration.
    
    DEPRECATED: This class is a compatibility wrapper around AdminStore and AdminAccessControl.
    New code should use those classes directly.
    """
    
    def __init__(self, profile_dir: Path):
        """
        Initialize AdminManager with configuration.
        
        Args:
            profile_dir: Path to the profile directory
        """
        self.profile_dir = Path(profile_dir)
        
        # Initialize config
        self.config = BotConfig(self.profile_dir)
        
        # Migrate password from env if not in config
        env_password = os.getenv("ADMIN_PASSWORD", "")
        if env_password and not self.config.admin_password:
            self.config.admin_password = env_password
        
        # We don't raise error if password is unset, instead we expect
        # user to set it via command or pre-configuration
        self.password = self.config.admin_password
        
        # Initialize new components
        self._admin_store = AdminStore(self.profile_dir)
        self._access_control = AdminAccessControl(self._admin_store, self.config)
    
    def _validate_admin_password(self, password: Optional[str] = None) -> bool:
        """
        Validate admin password if provided.
        
        Args:
            password: Password to validate (optional)
        
        Returns:
            True if password is valid or not required, False otherwise
        """
        if password is None:
            # Password validation is optional for backward compatibility
            return True
        return self.verify_password(password)
    
    def set_admin(self, user_id: int, username: str, first_name: str, last_name: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Set a user as admin.
        
        Args:
            user_id: Telegram user ID
            username: Telegram username
            first_name: User's first name
            last_name: User's last name (optional)
            password: Admin password for validation (optional)
        
        Returns:
            True if admin was set successfully
        """
        # Validate password if provided
        if not self._validate_admin_password(password):
            return False
        
        # Save admin info using AdminStore
        self._admin_store.set_admin(user_id, username, first_name, last_name)
        return True
    
    def get_admin(self) -> Optional[Dict]:
        """
        Get current admin info.
        
        Returns:
            Dict with admin info or None if no admin set
        """
        return self._access_control.get_admin()
    
    def _admin_exists(self) -> bool:
        """
        Check if admin exists.
        
        Returns:
            True if admin exists
        """
        return self._admin_store.admin_exists()
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin.
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            True if user is admin
        """
        return self._access_control.is_admin(user_id)
    
    def verify_password(self, password: str) -> bool:
        """
        Verify admin password.
        
        Args:
            password: Password to verify
        
        Returns:
            True if password is correct
        """
        return self._access_control.verify_password(password)
    
    def remove_admin(self) -> bool:
        """
        Remove current admin.
        
        Returns:
            True if admin was removed
        """
        return self._admin_store.remove_admin()
