"""LLM gateway for calling LLM with retry and error handling."""

from typing import List, Dict, Optional
from src.core.llm import LLMClient
from src.lib.syslog2 import *

# Import OpenAI exceptions for specific error handling
try:
    from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
except ImportError:
    # Fallback if openai is not available
    RateLimitError = type('RateLimitError', (Exception,), {})
    APIError = type('APIError', (Exception,), {})
    APIConnectionError = type('APIConnectionError', (Exception,), {})
    APITimeoutError = type('APITimeoutError', (Exception,), {})


class LLMGateway:
    """Gateway for LLM calls with retry logic and error handling."""
    
    def __init__(self, llm_client: LLMClient, prompt_builder, conversation_state, log_level: int = LOG_WARNING):
        """
        Initialize LLM gateway.
        
        Args:
            llm_client: LLMClient instance
            prompt_builder: PromptBuilder instance
            conversation_state: ConversationState instance
            log_level: Logging level
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.conversation_state = conversation_state
        self.log_level = log_level
    
    def _is_token_limit_error(self, error_msg: str) -> bool:
        """
        Check if error is related to token limit or payment issues.
        
        Args:
            error_msg: Error message string
            
        Returns:
            True if error is token limit related
        """
        error_lower = error_msg.lower()
        token_limit_keywords = [
            "402",
            "context_length_exceeded",
            "prompt tokens limit exceeded",
            "token limit exceeded",
            "maximum context length",
            "context length exceeded",
        ]
        return any(keyword in error_lower for keyword in token_limit_keywords)
    
    def _retry_after_reset(
        self, 
        context_chunks: List[Dict], 
        user_input: str, 
        system_prompt_template: Optional[str] = None
    ) -> str:
        """
        Retry LLM call after resetting context due to token limit error.
        
        Args:
            context_chunks: Retrieved context chunks
            user_input: User input message
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            Response from LLM or error message
        """
        if self.log_level <= LOG_INFO:
            syslog2(LOG_ERR, "token limit exceeded", action="resetting context and retrying")
        
        # Force reset context
        self.conversation_state.reset()
        
        # Reconstruct prompt without history using prompt_builder
        system_prompt, _ = self.prompt_builder.build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_input,
            custom_template=system_prompt_template,
            history=[]  # Empty history after reset
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        try:
            # Retry
            response = self.llm_client.complete(messages)
            return "Ошибка лимита токенов/баланса. Контекст сброшен.\n\n" + response
        except Exception as retry_e:
            syslog2(LOG_ERR, "retry failed", error=str(retry_e))
            from src.bot.utils.response_formatter import ResponseFormatter
            return ResponseFormatter.format_error_message("Не удалось получить ответ даже после сброса контекста (лимит токенов или баланс исчерпан).")
    
    def call_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        context_chunks: List[Dict], 
        user_input: str,
        system_prompt_template: Optional[str] = None
    ) -> str:
        """
        Call LLM with automatic retry on token limit errors.
        
        Args:
            messages: Messages to send to LLM
            context_chunks: Retrieved context chunks (for retry)
            user_input: User input (for retry)
            system_prompt_template: Optional custom template (for retry)
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If LLM call fails with non-token-limit error
        """
        try:
            return self.llm_client.complete(messages)
        except TimeoutError as e:
            # Handle timeout errors separately
            syslog2(LOG_ERR, "llm call timeout", error=str(e))
            raise
        except APITimeoutError as e:
            # Handle API timeout errors separately
            syslog2(LOG_ERR, "llm api timeout", error=str(e))
            raise
        except RateLimitError as e:
            # Handle rate limit errors separately
            syslog2(LOG_ERR, "llm rate limit error", error=str(e))
            raise
        except APIError as e:
            # Handle API errors separately
            syslog2(LOG_ERR, "llm api error", error=str(e))
            raise
        except APIConnectionError as e:
            # Handle connection errors separately
            syslog2(LOG_ERR, "llm connection error", error=str(e))
            raise
        except Exception as e:
            # Check for token limit or payment issues (existing logic)
            # This catches any other exceptions, including those that might be
            # wrapped or have token limit errors in their message
            error_msg = str(e)
            if self._is_token_limit_error(error_msg):
                return self._retry_after_reset(context_chunks, user_input, system_prompt_template)
            else:
                # Re-raise other errors
                raise
    
    def call_with_context(
        self,
        messages: List[Dict[str, str]],
        context_chunks: List[Dict],
        user_input: str,
        system_prompt_template: Optional[str] = None
    ) -> str:
        """
        Call LLM with context and handle errors.
        
        Args:
            messages: List of message dictionaries for LLM API
            context_chunks: List of context chunk dictionaries
            user_input: User query string
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If LLM call fails
        """
        response = self.call_with_retry(messages, context_chunks, user_input, system_prompt_template)
        return response

