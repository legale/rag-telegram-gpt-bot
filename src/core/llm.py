from openai import OpenAI
from typing import List, Dict, Optional, Generator
import os
import json

class LLMClient:
    """Client for interacting with LLM APIs (OpenRouter/OpenAI)."""
    
    def __init__(self, model: str = "openai/gpt-3.5-turbo", verbosity: int = 0):
        """
        Initialize the LLM client.
        
        Args:
            model: The model name to use (e.g., "openai/gpt-3.5-turbo", "anthropic/claude-3-opus").
            verbosity: Logging level (0=none, 1=info, 2=debug, 3=trace).
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set")
            
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model
        self.verbosity = verbosity
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generates a completion for the given messages.
        
        Args:
            messages: List of message dictionaries (role, content).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            The generated text response.
        """
        if self.verbosity >= 3:
            print(f"\n[LLM Request] Model: {self.model}")
            print(f"[LLM Request] Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            if self.verbosity >= 3:
                print(f"\n[LLM Response] Raw: {response}")
            
            return content if content else ""
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "I apologize, but I encountered an error while processing your request."

    def stream_complete(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream a completion from the LLM.
        
        Args:
            messages: List of message dictionaries.
            temperature: Sampling temperature.
            
        Yields:
            Chunks of generated text.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
