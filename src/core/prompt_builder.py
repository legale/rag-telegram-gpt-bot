"""Prompt builder for constructing LLM prompts."""

from typing import List, Dict, Optional, Tuple
from src.core.prompt import PromptEngine


class PromptBuilder:
    """Builds prompts for LLM calls with context and history."""
    
    def __init__(self, prompt_engine: PromptEngine, conversation_state, log_level: int = 4):
        """
        Initialize prompt builder.
        
        Args:
            prompt_engine: PromptEngine instance
            conversation_state: ConversationState instance
            log_level: Logging level
        """
        self.prompt_engine = prompt_engine
        self.conversation_state = conversation_state
        self.log_level = log_level
    
    def build_history_for_prompt(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """
        Build history for prompt from chat history.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of history entries with sender and content
        """
        history_for_prompt = []
        for msg in self.conversation_state.get_history(max_messages):
            sender = "User" if msg["role"] == "user" else "Bot"
            history_for_prompt.append(
                {"sender": sender, "content": msg["content"]}
            )
        return history_for_prompt
    
    def build_prompt_and_history(
        self, 
        context_chunks: List[Dict], 
        user_task: str, 
        custom_template: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Build system prompt and history for prompt (helper to reduce duplication).
        
        Args:
            context_chunks: Retrieved context chunks
            user_task: User task/query string
            custom_template: Optional custom system prompt template
            history: Optional pre-built history (if None, builds from chat_history)
            
        Returns:
            Tuple of (system_prompt, history_for_prompt)
        """
        if history is None:
            max_messages = 5  # Default value from config
            history_for_prompt = self.build_history_for_prompt(max_messages=max_messages)
        else:
            history_for_prompt = history
        
        system_prompt = self.prompt_engine.construct_prompt(
            context_chunks=context_chunks,
            chat_history=history_for_prompt,
            user_task=user_task,
            custom_template=custom_template,
            log_level=self.log_level
        )
        return system_prompt, history_for_prompt
    
    def build_messages_for_llm(
        self,
        context_chunks: List[Dict],
        user_input: str,
        system_prompt_template: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages list for LLM API call.
        
        Args:
            context_chunks: List of context chunk dictionaries
            user_input: User query string
            system_prompt_template: Optional custom system prompt template
            
        Returns:
            List of message dictionaries for LLM API
        """
        # системный промпт: контекст + история + инструкции, но без дублирования user_input
        system_prompt, _ = self.build_prompt_and_history(
            context_chunks=context_chunks,
            user_task=user_input,
            custom_template=system_prompt_template
        )

        if self.log_level <= 7:  # LOG_DEBUG
            from src.lib.syslog2 import syslog2, LOG_DEBUG
            syslog2(LOG_DEBUG, "system prompt constructed", length=len(system_prompt))

        from src.lib.syslog2 import syslog2, LOG_NOTICE
        syslog2(LOG_NOTICE, "querying llm")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        return messages

