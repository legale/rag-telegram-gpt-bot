"""
Admin management utilities for Legale Bot.
Handles admin authentication and storage.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict


from src.bot.config import BotConfig

class AdminManager:
    """Manages bot administrators and configuration."""
    
    def __init__(self, profile_dir: Path):
        """
        Initialize AdminManager with configuration.
        
        Args:
            profile_dir: Path to the profile directory
        """
        self.profile_dir = Path(profile_dir)
        self.admin_file = self.profile_dir / "admin.json"
        
        # Initialize config
        self.config = BotConfig(self.profile_dir)
        
        # Migrate password from env if not in config
        env_password = os.getenv("ADMIN_PASSWORD", "")
        if env_password and not self.config.admin_password:
            self.config.admin_password = env_password
        
        # We don't raise error if password is unset, instead we expect
        # user to set it via command or pre-configuration
        self.password = self.config.admin_password
    
    def _load_admin_data(self) -> Dict:
        """Load admin data from file."""
        if not self.admin_file.exists():
            return {}
        
        try:
            with open(self.admin_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_admin_data(self, data: Dict):
        """Save admin data to file."""
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.admin_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Restrict file permissions (owner read/write only)
        os.chmod(self.admin_file, 0o600)
    
    def set_admin(self, user_id: int, username: str, first_name: str, last_name: Optional[str] = None) -> bool:
        """
        Set a user as admin.
        
        Args:
            user_id: Telegram user ID
            username: Telegram username
            first_name: User's first name
            last_name: User's last name (optional)
        
        Returns:
            True if admin was set successfully
        """
        data = {
            'user_id': user_id,
            'username': username,
            'first_name': first_name,
            'last_name': last_name or '',
            'full_name': f"{first_name} {last_name}".strip() if last_name else first_name
        }
        
        self._save_admin_data(data)
        return True
    
    def get_admin(self) -> Optional[Dict]:
        """
        Get current admin info.
        
        Returns:
            Dict with admin info or None if no admin set
        """
        data = self._load_admin_data()
        return data if data else None
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin.
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            True if user is admin
        """
        admin = self.get_admin()
        return admin is not None and admin.get('user_id') == user_id
    
    def verify_password(self, password: str) -> bool:
        """
        Verify admin password.
        
        Args:
            password: Password to verify
        
        Returns:
            True if password is correct
        """
        return password == self.config.admin_password
    
    def remove_admin(self) -> bool:
        """
        Remove current admin.
        
        Returns:
            True if admin was removed
        """
        if self.admin_file.exists():
            self.admin_file.unlink()
            return True
        return False
