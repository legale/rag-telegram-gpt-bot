"""App class - main application entry point."""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from src.bot.core import LegaleBot
from src.bot.admin import AdminManager
from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.app.main_cli import handle_command
from src.app.types import AppRequest, AppResponse
from src.core.command_service import CommandService
from src.lib.syslog2 import *


class App:
    """Main application class that provides unified access to dependencies."""
    
    def __init__(
        self,
        bot: LegaleBot,
        command_service: CommandService,
        admin_manager: Optional[AdminManager] = None,
        debug_rag: bool = False
    ):
        """
        Initialize App instance.
        
        Args:
            bot: LegaleBot instance
            command_service: CommandService instance with registered commands
            admin_manager: Optional AdminManager instance
            debug_rag: Whether to enable debug RAG mode
        """
        self.bot = bot
        self.command_service = command_service
        self.admin_manager = admin_manager
        self.debug_rag = debug_rag
    
    @property
    def dispatcher(self):
        """Get CommandDispatcher instance from CommandService (for backward compatibility)."""
        return self.command_service.dispatcher
    
    def handle_command(
        self,
        command: str,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Handle a command string.
        
        Args:
            command: Command string (e.g., "/find 2.0 vpn туннель" or "/help")
            user_id: Optional user ID
            chat_id: Optional chat ID
            
        Returns:
            Response string if command was handled, None if not a command
        """
        return handle_command(
            command=command,
            dispatcher=self.command_service.dispatcher,
            user_id=user_id,
            chat_id=chat_id
        )
    
    def get_database(self) -> Database:
        """
        Get Database instance from bot.
        
        Returns:
            Database instance
        """
        return self.bot.db
    
    def get_vector_store(self) -> VectorStore:
        """
        Get VectorStore instance from bot.
        
        Returns:
            VectorStore instance
        """
        return self.bot.vector_store
    
    def handle_request(self, request: AppRequest) -> AppResponse:
        """
        Handle a request from transport layer.
        
        Args:
            request: AppRequest with user input
            
        Returns:
            AppResponse with text and optional actions
        """
        text = request.text.strip()
        if not text:
            return AppResponse(text="")
        
        # Check if it's a command
        command_response = self.handle_command(
            command=text,
            user_id=request.user_id,
            chat_id=request.chat_id
        )
        
        if command_response is not None:
            # Command was handled
            return AppResponse(text=command_response)
        
        # Not a command - handle as regular query
        try:
            # Get chunks from meta if available
            chunks = request.meta.get("chunks", 5) if request.meta else 5
            response_text = self.bot.chat(text, n_results=chunks)
            return AppResponse(text=response_text)
        except Exception as e:
            syslog2(LOG_ERR, "chat error", error=str(e))
            return AppResponse(text=f"Ошибка при обработке запроса: {e}")
