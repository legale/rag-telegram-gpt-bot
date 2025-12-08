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
        """Load configuration from file or return defaults. Auto-save missing defaults."""
        defaults = {
            "admin_password": "",
            "allowed_chats": [],
            "response_frequency": 0,
            "system_prompt": "",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_generator": "local"
        }
        
        if not self.config_file.exists():
            # Create file with defaults
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            os.chmod(self.config_file, 0o600)
            return defaults
            
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            # Check if any defaults are missing and add them
            updated = False
            for key, default_value in defaults.items():
                if key not in data:
                    data[key] = default_value
                    updated = True
            
            # Save updated config if defaults were added
            if updated:
                with open(self.config_file, 'w') as f:
                    json.dump(data, f, indent=2)
                os.chmod(self.config_file, 0o600)
            
            return data
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, create new one with defaults
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            os.chmod(self.config_file, 0o600)
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
        return self.data.get("response_frequency", 0)
        
    @response_frequency.setter
    def response_frequency(self, value: int):
        if value < 0:
            value = 0
        self.data["response_frequency"] = value
        self.save()

    @property
    def current_model(self) -> str:
        return self.data.get("current_model", "")

    @current_model.setter
    def current_model(self, value: str):
        self.data["current_model"] = value
        self.save()

    @property
    def system_prompt(self) -> str:
        return self.data.get("system_prompt", "")

    @system_prompt.setter
    def system_prompt(self, value: str):
        self.data["system_prompt"] = value
        self.save()

    @property
    def embedding_model(self) -> str:
        return self.data.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")

    @embedding_model.setter
    def embedding_model(self, value: str):
        self.data["embedding_model"] = value
        self.save()

    @property
    def embedding_generator(self) -> str:
        return self.data.get("embedding_generator", "local")

    @embedding_generator.setter
    def embedding_generator(self, value: str):
        if value.lower() not in ["openrouter", "openai", "local"]:
            raise ValueError(f"embedding_generator must be one of: openrouter, openai, local")
        self.data["embedding_generator"] = value.lower()
        self.save()

