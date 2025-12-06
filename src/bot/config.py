"""
Configuration management for Legale Bot profiles.
Handles settings like admin password, allowed chats, and response frequency.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union

class BotConfig:
    """Manages profile-specific configuration stored in config.json."""
    
    def __init__(self, profile_dir: Path):
        """
        Initialize BotConfig.
        
        Args:
            profile_dir: Path to the profile directory
        """
        self.profile_dir = Path(profile_dir)
        self.config_file = self.profile_dir / "config.json"
        self.data = self._load()
        
    def _load(self) -> Dict:
        """Load configuration from file or return defaults."""
        defaults = {
            "admin_password": "",
            "allowed_chats": [],
            "response_frequency": 1
        }
        
        if not self.config_file.exists():
            return defaults
            
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**defaults, **data}
        except (json.JSONDecodeError, IOError):
            return defaults

    def save(self):
        """Save configuration to file."""
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Restrict permissions (sensitive data included)
        os.chmod(self.config_file, 0o600)
    
    @property
    def admin_password(self) -> str:
        return self.data.get("admin_password", "")
    
    @admin_password.setter
    def admin_password(self, value: str):
        self.data["admin_password"] = value
        self.save()

    @property
    def allowed_chats(self) -> List[int]:
        return self.data.get("allowed_chats", [])

    @allowed_chats.setter
    def allowed_chats(self, value: List[int]):
        self.data["allowed_chats"] = value
        self.save()
        
    def add_allowed_chat(self, chat_id: int):
        if chat_id not in self.allowed_chats:
            chats = self.allowed_chats
            chats.append(chat_id)
            self.allowed_chats = chats
            
    def remove_allowed_chat(self, chat_id: int):
        if chat_id in self.allowed_chats:
            chats = self.allowed_chats
            chats.remove(chat_id)
            self.allowed_chats = chats

    @property
    def response_frequency(self) -> int:
        return self.data.get("response_frequency", 1)
        
    @response_frequency.setter
    def response_frequency(self, value: int):
        if value < 1:
            value = 1
        self.data["response_frequency"] = value
        self.save()
