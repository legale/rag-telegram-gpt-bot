"""Command service for unified command registration and dispatch."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from src.core.dispatcher import (
    AsyncCommandHandler,
    CommandContext,
    CommandDispatcher,
    CommandHandler,
    CommandResult,
)

if TYPE_CHECKING:
    from src.app.types import CommandRequest


class CommandService:
    """
    Unified command service with explicit registry.
    
    This service provides a single point for command registration,
    replacing the need for multiple dispatcher creation functions.
    """
    
    def __init__(self):
        """Initialize command service with empty dispatcher."""
        self.dispatcher = CommandDispatcher()
        self._registry: Dict[str, str] = {}  # command_name -> handler_type ("sync" or "async")
    
    def register(self, command_name: str, handler: CommandHandler) -> None:
        """
        Register a synchronous command handler.
        
        Args:
            command_name: Command name (e.g., "start", "help", "find")
            handler: CommandHandler instance
        """
        self.dispatcher.register(command_name, handler)
        self._registry[command_name] = "sync"
    
    def register_async(self, command_name: str, handler: AsyncCommandHandler) -> None:
        """
        Register an asynchronous command handler.
        
        Args:
            command_name: Command name (e.g., "admin", "admin_set")
            handler: AsyncCommandHandler instance
        """
        self.dispatcher.register_async(command_name, handler)
        self._registry[command_name] = "async"
    
    def dispatch(self, request: CommandContext | "CommandRequest") -> CommandResult:
        """
        Dispatch a command to its handler.
        
        Args:
            request: CommandContext or CommandRequest
            
        Returns:
            CommandResult from handler, or error result if command not found
        """
        context = self._to_command_context(request)
        return self.dispatcher.dispatch(context)

    def _to_command_context(self, request: Any) -> CommandContext:
        if isinstance(request, CommandContext):
            return request

        try:
            from src.app.types import CommandRequest as AppCommandRequest
        except Exception:
            AppCommandRequest = None

        if AppCommandRequest is not None and isinstance(request, AppCommandRequest):
            command_name = (request.name or "").strip()
            if not command_name and request.raw:
                command_name = request.raw.strip().split(maxsplit=1)[0]

            metadata: dict = {}
            if request.meta:
                metadata.update(request.meta)
            if request.raw:
                metadata.setdefault("raw", request.raw)

            return CommandContext(
                user_id=request.user_id,
                chat_id=request.chat_id,
                command_name=command_name,
                args=list(request.args or []),
                metadata=metadata,
            )

        raise TypeError("dispatch expects CommandContext or CommandRequest")
    
    async def dispatch_async(self, context: CommandContext) -> CommandResult:
        """
        Dispatch a command to its async handler.
        
        Args:
            context: Command context with command name and arguments
            
        Returns:
            CommandResult from handler, or error result if command not found
        """
        return await self.dispatcher.dispatch_async(context)
    
    def get_registry(self) -> Dict[str, str]:
        """
        Get registry of registered commands.
        
        Returns:
            Dictionary mapping command names to handler types
        """
        return self._registry.copy()
