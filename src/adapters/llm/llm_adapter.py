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
        # Convert to messages format expected by LLMClient
        messages: List[Dict[str, str]] = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        # Extract temperature and max_tokens from kwargs if provided
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1500)
        
        return self.llm_client.complete(messages, temperature=temperature, max_tokens=max_tokens)

