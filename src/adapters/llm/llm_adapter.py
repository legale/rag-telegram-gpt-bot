"""LLM adapter implementing LLM interface."""

from __future__ import annotations

from typing import List, Dict, Optional

from src.core.interfaces import LLM
from src.core.llm import LLMClient


class LLMAdapter:
    """Adapter wrapping LLMClient to implement LLM interface."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the adapter with an LLMClient instance.

        Args:
            llm_client: LLMClient instance from src.core.llm
        """
        self.llm_client = llm_client

    def _build_messages(self, prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build messages list from prompt and optional system message.
        
        Args:
            prompt: User prompt text
            system: Optional system message
            
        Returns:
            List of message dictionaries in format expected by LLMClient
        """
        messages: List[Dict[str, str]] = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _extract_kwargs(self, kwargs: dict) -> tuple[float, int]:
        """
        Extract temperature and max_tokens from kwargs with defaults.
        
        Args:
            kwargs: Dictionary with optional temperature and max_tokens
            
        Returns:
            Tuple of (temperature, max_tokens)
        """
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1500)
        return temperature, max_tokens
    
    def _call_llm(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """
        Call LLM client with messages and parameters.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        return self.llm_client.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt text
            system: Optional system message
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        messages = self._build_messages(prompt, system)
        temperature, max_tokens = self._extract_kwargs(kwargs)
        return self._call_llm(messages, temperature, max_tokens)

