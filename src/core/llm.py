from openai import OpenAI
from typing import List, Dict, Optional, Generator
import os
import json
import tiktoken
from src.core.syslog2 import *

class LLMClient:
    """Client for interacting with LLM APIs (OpenRouter/OpenAI)."""
    
    def __init__(self, model: str, log_level: int = LOG_WARNING):
        """
        Initialize the LLM client.
        
        Args:
            model: The model name to use (e.g., "openai/gpt-oss-20b:free", "anthropic/claude-3-opus").
            log_level: Logging level (LOG_ALERT=1, LOG_CRIT=2, LOG_ERR=3, LOG_WARNING=4, LOG_NOTICE=5, LOG_INFO=6, LOG_DEBUG=7).
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set")
            
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model
        self.log_level = log_level
        

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # Initialize tokenizer for token counting
        try:
            # Extract base model name for tiktoken (remove provider prefix)
            base_model = model.split('/')[-1] if '/' in model else model
            self.encoding = tiktoken.encoding_for_model(base_model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by gpt-3.5-turbo and gpt-4)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Control HTTP logging based on log_level
        import logging
        if log_level >= LOG_WARNING:
            # Completely silence HTTP logging at LOG_WARNING and above (less verbose)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
        elif log_level <= LOG_DEBUG:
            # Enable low-level HTTP logging only at LOG_DEBUG and below (more verbose)
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.DEBUG)
            httpx_logger.propagate = True

    @property
    def model_name(self) -> str:
        return self.model
    
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count the number of tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries (role, content).
            
        Returns:
            Total number of tokens.
        """
        num_tokens = 0
        for message in messages:
            # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            num_tokens += 4  # message overhead
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(str(value)))
        num_tokens += 2  # every reply is primed with <|start|>assistant
        return num_tokens

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """
        Generates a completion for the given messages.
        
        Args:
            messages: List of message dictionaries (role, content).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            The generated text response.
        """
        if self.log_level <= LOG_INFO:
            syslog2(LOG_DEBUG, "LLM request", model=self.model)
            # Log full messages only at LOG_DEBUG to avoid spam, or truncating
            if self.log_level <= LOG_DEBUG:
                 syslog2(LOG_DEBUG, "LLM messages", messages=messages)
        
        # Log input messages at LOG_INFO level
        if self.log_level <= LOG_INFO:
            syslog2(LOG_INFO, "LLM input", messages=messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            if self.log_level <= LOG_INFO:
                syslog2(LOG_DEBUG, "LLM response", response=response)
            
            # Log output response at LOG_INFO level
            if self.log_level <= LOG_INFO:
                syslog2(LOG_INFO, "LLM output", response=content)
            
            if not content:
                finish_reason = response.choices[0].finish_reason
                syslog2(LOG_WARNING, "llm returned empty content", model=self.model, finish_reason=finish_reason)
                
            return content if content else ""
        except Exception as e:
            # Re-raise exception to be handled by caller (LegaleBot)
            raise e

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
