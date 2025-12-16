"""Conversation state management for chat sessions."""

from typing import List, Dict, Optional


class ConversationState:
    """Manages conversation state including chat history and RAG context cache."""
    
    def __init__(self):
        """Initialize empty conversation state."""
        # Simple in-memory history for the current session
        self.chat_history: List[Dict[str, str]] = []
        
        # RAG context cache for reuse across messages
        self.active_context_chunks: Optional[List[Dict]] = None
        self.active_context_query: Optional[str] = None
        self.active_context_score: Optional[float] = None
    
    def reset(self) -> None:
        """Reset conversation state (clear history and context cache)."""
        self.chat_history = []
        self.clear_active_context("manual_reset")
    
    def clear_active_context(self, reason: str) -> None:
        """
        Clear the active RAG context cache.
        
        Args:
            reason: Reason for clearing (for logging)
        """
        self.active_context_chunks = None
        self.active_context_query = None
        self.active_context_score = None
    
    def add_user_message(self, content: str) -> None:
        """Add user message to chat history."""
        self.chat_history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to chat history."""
        self.chat_history.append({"role": "assistant", "content": content})
    
    def get_history(self, max_messages: int = 5) -> List[Dict[str, str]]:
        """
        Get recent chat history.
        
        Args:
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.chat_history[-max_messages:] if self.chat_history else []
    
    def get_last_user_message(self) -> Optional[str]:
        """Get last user message from chat history."""
        for msg in reversed(self.chat_history):
            if msg["role"] == "user":
                return msg["content"]
        return None

