"""
Configuration management for Legale Bot profiles.
Handles settings like admin password, allowed chats, and response frequency.

DEPRECATED: This module is deprecated. Use src.app.config_store.BotConfig instead.
This module is kept for backward compatibility and will be removed in a future version.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List


class BotConfig:
    """
    Deprecated: Use src.app.config_store.BotConfig instead.
    
    This class is kept for backward compatibility and will be removed in a future version.
    Manages profile-specific configuration stored in config.json.
    """
    
    def __init__(self, profile_dir: Path):
        warnings.warn(
            "src.bot.config.BotConfig is deprecated. Use src.app.config_store.BotConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.profile_dir = Path(profile_dir)
        self.config_file = self.profile_dir / "config.json"
        self.data = self._load()
        
    def _get_defaults(self) -> Dict:
        """Get default configuration values."""
        return {
            "admin_password": "",
            "allowed_chats": [],
            "response_frequency": 0,
            "system_prompt": "",
            "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
            "embedding_generator": "local",
            "current_model": "openai/gpt-oss-20b:free",
            "only_unnamed": True,
            "rebuild": False,
            "chunk_token_min": 50,
            "chunk_token_max": 1024,
            "chunk_overlap_ratio": 0.30,
            "cosine_distance_thr": 4,
            "rag_ntop": 40,
            "fts5_score_thr": 0.2,
            "profile_context_tokens": 60000,
            "llm_max_tokens": 60000
        }

    def _create_default_config(self) -> Dict:
        """
        Create config file with default values.
        
        Returns:
            Dictionary with default configuration values
        """
        defaults = self._get_defaults()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(defaults, f, indent=2)
        os.chmod(self.config_file, 0o600)
        return defaults

    def _load_existing_config(self) -> Dict:
        """
        Load existing configuration from file.
        
        Returns:
            Dictionary with configuration data
            
        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            IOError: If file cannot be read
        """
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def _add_missing_defaults(self, data: Dict) -> bool:
        """
        Add missing default values to configuration data.
        
        Args:
            data: Configuration dictionary to update
            
        Returns:
            True if any defaults were added, False otherwise
        """
        defaults = self._get_defaults()
        updated = False
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
                updated = True
        return updated

    def _load(self) -> Dict:
        """Load configuration from file or return defaults. Auto-save missing defaults."""
        if not self.config_file.exists():
            return self._create_default_config()
            
        try:
            data = self._load_existing_config()
            
            # Check if any defaults are missing and add them
            if self._add_missing_defaults(data):
                # Save updated config if defaults were added
                with open(self.config_file, 'w') as f:
                    json.dump(data, f, indent=2)
                os.chmod(self.config_file, 0o600)
            
            return data
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, create new one with defaults
            return self._create_default_config()

    def _ensure_profile_dir(self):
        """Ensure profile directory exists."""
        self.profile_dir.mkdir(parents=True, exist_ok=True)
    
    def _write_config_file(self):
        """Write configuration data to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def _set_file_permissions(self):
        """Set restrictive file permissions (sensitive data included)."""
        os.chmod(self.config_file, 0o600)
    
    def save(self):
        """Save configuration to file."""
        self._ensure_profile_dir()
        self._write_config_file()
        self._set_file_permissions()
    
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
        
    def _validate_response_frequency(self, value: int) -> int:
        """
        Validate response_frequency value.
        
        Args:
            value: Response frequency value
            
        Returns:
            Validated value (clamped to >= 0)
        """
        if value < 0:
            return 0
        return value
    
    @response_frequency.setter
    def response_frequency(self, value: int):
        value = self._validate_response_frequency(value)
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
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt from config, or return default template if empty.
        
        Returns:
            System prompt string (custom from config or default template)
        """
        prompt = self.system_prompt
        if not prompt:
            from src.core.prompt import PromptEngine
            return PromptEngine.SYSTEM_PROMPT_TEMPLATE
        return prompt

    @property
    def embedding_model(self) -> str:
        return self.data.get("embedding_model", "paraphrase-multilingual-mpnet-base-v2")

    @embedding_model.setter
    def embedding_model(self, value: str):
        self.data["embedding_model"] = value
        self.save()

    @property
    def embedding_generator(self) -> str:
        return self.data.get("embedding_generator", "local")

    def _validate_embedding_generator(self, value: str) -> str:
        """
        Validate embedding_generator value.
        
        Args:
            value: Embedding generator name
            
        Returns:
            Validated lowercase value
            
        Raises:
            ValueError: If value is not one of allowed generators
        """
        if value.lower() not in ["openrouter", "openai", "local"]:
            raise ValueError(f"embedding_generator must be one of: openrouter, openai, local")
        return value.lower()
    
    @embedding_generator.setter
    def embedding_generator(self, value: str):
        value = self._validate_embedding_generator(value)
        self.data["embedding_generator"] = value
        self.save()
    
    @property
    def chunk_token_min(self) -> int:
        return self.data.get("chunk_token_min", 50)
    
    def _validate_chunk_token_min(self, value: int) -> int:
        """
        Validate chunk_token_min value.
        
        Args:
            value: Minimum chunk token count
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value < 1:
            raise ValueError("chunk_token_min must be a positive integer")
        return value
    
    @chunk_token_min.setter
    def chunk_token_min(self, value: int):
        value = self._validate_chunk_token_min(value)
        self.data["chunk_token_min"] = value
        self.save()

    @property
    def chunk_token_max(self) -> int:
        return self.data.get("chunk_token_max", 400)
    
    def _validate_chunk_token_max(self, value: int) -> int:
        """
        Validate chunk_token_max value.
        
        Args:
            value: Maximum chunk token count
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value < 1:
            raise ValueError("chunk_token_max must be a positive integer")
        return value
    
    @chunk_token_max.setter
    def chunk_token_max(self, value: int):
        value = self._validate_chunk_token_max(value)
        self.data["chunk_token_max"] = value
        self.save()

    @property
    def chunk_overlap_ratio(self) -> float:
        return self.data.get("chunk_overlap_ratio", 0.3)
    
    def _validate_chunk_overlap_ratio(self, value: float) -> float:
        """
        Validate chunk_overlap_ratio value.
        
        Args:
            value: Chunk overlap ratio (0.0-1.0)
            
        Returns:
            Validated float value
            
        Raises:
            ValueError: If value is not a float between 0 and 1
        """
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError("chunk_overlap_ratio must be a float between 0 and 1")
        return float(value)
    
    @chunk_overlap_ratio.setter
    def chunk_overlap_ratio(self, value: float):
        value = self._validate_chunk_overlap_ratio(value)
        self.data["chunk_overlap_ratio"] = value
        self.save()

    @property
    def cosine_distance_thr(self) -> float:
        return self.data.get("cosine_distance_thr", 1.5)
    
    def _validate_cosine_distance_thr(self, value: float) -> float:
        """
        Validate cosine_distance_thr value.
        
        Args:
            value: Cosine distance threshold
            
        Returns:
            Validated float value
            
        Raises:
            ValueError: If value is not a non-negative float
        """
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("cosine_distance_thr must be a non-negative float")
        return float(value)
    
    @cosine_distance_thr.setter
    def cosine_distance_thr(self, value: float):
        value = self._validate_cosine_distance_thr(value)
        self.data["cosine_distance_thr"] = value
        self.save()

    @property
    def rag_ntop(self) -> int:
        return self.data.get("rag_ntop", 0)
    
    def _validate_rag_ntop(self, value: int) -> int:
        """
        Validate rag_ntop value.
        
        Args:
            value: RAG top N value
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If value is not a non-negative integer
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("rag_ntop must be a non-negative integer")
        return value
    
    @rag_ntop.setter
    def rag_ntop(self, value: int):
        value = self._validate_rag_ntop(value)
        self.data["rag_ntop"] = value
        self.save()

    @property
    def fts5_score_thr(self) -> float:
        return self.data.get("fts5_score_thr", 0.2)
    
    def _validate_fts5_score_thr(self, value: float) -> float:
        """
        Validate fts5_score_thr value.
        
        Args:
            value: FTS5 score threshold (0.0-1.0)
            
        Returns:
            Validated float value
            
        Raises:
            ValueError: If value is not a float between 0 and 1
        """
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError("fts5_score_thr must be a float between 0 and 1")
        return float(value)
    
    @fts5_score_thr.setter
    def fts5_score_thr(self, value: float):
        value = self._validate_fts5_score_thr(value)
        self.data["fts5_score_thr"] = value
        self.save()

    @property
    def profile_context_tokens(self) -> int:
        return self.data.get("profile_context_tokens", 60000)
    
    def _validate_profile_context_tokens(self, value: int) -> int:
        if not isinstance(value, int) or value < 1000:
            raise ValueError("profile_context_tokens must be a positive integer >= 1000")
        return value
        
    @profile_context_tokens.setter
    def profile_context_tokens(self, value: int):
        value = self._validate_profile_context_tokens(value)
        self.data["profile_context_tokens"] = value
        self.save()

    @property
    def llm_max_tokens(self) -> int:
        return self.data.get("llm_max_tokens", 0)

    def _validate_llm_max_tokens(self, value: int) -> int:
        if not isinstance(value, int) or value < 0:
            raise ValueError("llm_max_tokens must be a non-negative integer")
        return value

    @llm_max_tokens.setter
    def llm_max_tokens(self, value: int):
        value = self._validate_llm_max_tokens(value)
        self.data["llm_max_tokens"] = value
        self.save()
