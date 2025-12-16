"""Command service for unified command registration and dispatch."""

from __future__ import annotations

from typing import Optional, Dict
from src.core.dispatcher import CommandDispatcher, CommandHandler, AsyncCommandHandler, CommandContext, CommandResult


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
    
    def dispatch(self, context: CommandContext) -> CommandResult:
        """
        Dispatch a command to its handler.
        
        Args:
            context: Command context with command name and arguments
            
        Returns:
            CommandResult from handler, or error result if command not found
        """
        return self.dispatcher.dispatch(context)
    
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

