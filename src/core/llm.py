from openai import OpenAI
from typing import List, Dict, Optional, Generator, Union
import os
import json
import tiktoken
import time
from src.lib.syslog2 import *

class LLMClient:
    """Client for interacting with LLM APIs (OpenRouter/OpenAI)."""
    
    def __init__(self, model: str, log_level: int = LOG_WARNING):
        """
        Initialize the LLM client.
        
        Args:
            model: The model name to use (e.g., "openai/gpt-oss-20b:free", "anthropic/claude-3-opus").
            log_level: syslog2 log level (LOG_WARNING=4, LOG_INFO=6, LOG_DEBUG=7).
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
        if log_level < LOG_INFO:
            # Completely silence HTTP logging below LOG_INFO
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
        elif log_level >= LOG_DEBUG:
            # Enable low-level HTTP logging at LOG_DEBUG and above
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
            num_tokens += self._count_message_tokens(message)
        num_tokens += 2  # every reply is primed with <|start|>assistant
        return num_tokens
    
    def _count_message_tokens(self, message: Dict[str, str]) -> int:
        """
        Count the number of tokens for a single message.
        
        Args:
            message: Message dictionary (role, content).
            
        Returns:
            Number of tokens for this message.
        """
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        num_tokens = 4  # message overhead
        for key, value in message.items():
            num_tokens += len(self.encoding.encode(str(value)))
        return num_tokens

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if error is retryable (rate limit or temporary network error).
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable, False otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Rate limit errors
        if "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str:
            return True
        
        # Network/timeout errors
        if isinstance(error, (TimeoutError, OSError)):
            return True
        
        # Connection errors
        if "connection" in error_str or "timeout" in error_str or "network" in error_str:
            return True
        
        # API errors that might be temporary
        if error_type in ("APIConnectionError", "APITimeoutError", "RateLimitError"):
            return True

        return False

    def complete(self, prompt_or_messages: Union[str, List[Dict[str, str]]], system: Optional[str] = None, **kwargs) -> str:
        """
        Complete text using LLM. Supports both prompt string (protocol compliant) and messages list (backward compat).
        
        Args:
            prompt_or_messages: Prompt string OR List of message dictionaries
            system: Optional system message (only used if prompt_or_messages is str)
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        if isinstance(prompt_or_messages, str):
            # Protocol compliant mode
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt_or_messages})
        else:
            # Backward compatible mode
            messages = prompt_or_messages
            
        # Extract params from kwargs if present, otherwise use defaults
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1500)
        
        return self.complete_messages(messages, temperature=temperature, max_tokens=max_tokens)

    
    def complete_messages(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """
        Generates a completion for the given messages with retry logic.
        
        Args:
            messages: List of message dictionaries (role, content).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            The generated text response.
        """
        # Log LLM input at LOG_INFO level
        if self.log_level <= LOG_INFO:
            syslog2(LOG_INFO, "llm input", model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        
        if self.log_level >= LOG_DEBUG:
            syslog2(LOG_DEBUG, "LLM request", model=self.model)
            # Log full messages at LOG_DEBUG level
            syslog2(LOG_DEBUG, "LLM messages", messages=messages)

        max_retries = 3
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30.0
                )
                
                content = response.choices[0].message.content
                
                # Log LLM output at LOG_INFO level
                if self.log_level <= LOG_INFO:
                    syslog2(LOG_INFO, "llm output", model=self.model, response=content)
                
                if self.log_level >= LOG_DEBUG:
                    syslog2(LOG_DEBUG, "LLM response", response=response)
                
                if not content:
                    finish_reason = response.choices[0].finish_reason
                    syslog2(LOG_WARNING, "llm returned empty content", model=self.model, finish_reason=finish_reason)
                    
                return content if content else ""
                
            except Exception as e:
                # Check if error is retryable
                if attempt < max_retries - 1 and self._is_retryable_error(e):
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    if self.log_level <= LOG_WARNING:
                        syslog2(LOG_WARNING, "llm retry", 
                               model=self.model, 
                               attempt=attempt + 1, 
                               max_retries=max_retries,
                               error=str(e),
                               delay=delay)
                    time.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries reached - re-raise
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
            stream=True,
            timeout=30.0
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
