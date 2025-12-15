"""Command dispatcher for routing commands to use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


@dataclass
class CommandContext:
    """Context for command execution."""
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    command_name: str = ""
    args: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class CommandHandler(ABC):
    """Abstract base class for command handlers."""

    @abstractmethod
    def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle a command.

        Args:
            context: Command context with user info and arguments

        Returns:
            CommandResult with success status and message
        """
        pass


class AsyncCommandHandler(ABC):
    """Abstract base class for async command handlers."""

    @abstractmethod
    async def handle(self, context: CommandContext) -> CommandResult:
        """
        Handle a command asynchronously.

        Args:
            context: Command context with user info and arguments

        Returns:
            CommandResult with success status and message
        """
        pass


class CommandDispatcher:
    """
    Dispatches commands to appropriate handlers.

    This class provides a centralized way to route commands to their handlers,
    making it easier to add new commands and test command handling logic.
    Supports both synchronous and asynchronous handlers.
    """

    def __init__(self):
        """Initialize dispatcher with empty handler registry."""
        self.handlers: Dict[str, CommandHandler] = {}
        self.async_handlers: Dict[str, AsyncCommandHandler] = {}

    def register(self, command_name: str, handler: CommandHandler) -> None:
        """
        Register a synchronous command handler.

        Args:
            command_name: Command name (e.g., "start", "help", "find")
            handler: CommandHandler instance
        """
        # Normalize command name (remove leading slash, lowercase)
        normalized = command_name.lstrip("/").lower()
        self.handlers[normalized] = handler

    def register_async(self, command_name: str, handler: AsyncCommandHandler) -> None:
        """
        Register an asynchronous command handler.

        Args:
            command_name: Command name (e.g., "admin", "admin_set")
            handler: AsyncCommandHandler instance
        """
        # Normalize command name (remove leading slash, lowercase)
        normalized = command_name.lstrip("/").lower()
        self.async_handlers[normalized] = handler

    def _normalize_command_name(self, command_name: str) -> str:
        """
        Normalize command name by removing leading slash and converting to lowercase.
        
        Args:
            command_name: Raw command name
            
        Returns:
            Normalized command name
        """
        return command_name.lstrip("/").lower()
    
    def _find_handler(self, command_name: str) -> Optional[CommandHandler]:
        """
        Find handler for normalized command name.
        
        Args:
            command_name: Normalized command name
            
        Returns:
            CommandHandler instance or None if not found
        """
        return self.handlers.get(command_name)
    
    def _execute_handler(self, handler: CommandHandler, context: CommandContext) -> CommandResult:
        """
        Execute handler and handle exceptions.
        
        Args:
            handler: CommandHandler instance
            context: Command context
            
        Returns:
            CommandResult from handler or error result if exception occurred
        """
        try:
            return handler.handle(context)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении команды: {e}",
                error=str(e)
            )
    
    def dispatch(self, context: CommandContext) -> CommandResult:
        """
        Dispatch a command to its handler.

        Args:
            context: Command context with command name and arguments

        Returns:
            CommandResult from handler, or error result if command not found
        """
        # Normalize command name
        command_name = self._normalize_command_name(context.command_name)

        # Find handler
        handler = self._find_handler(command_name)

        if not handler:
            return CommandResult(
                success=False,
                message=f"Неизвестная команда: /{command_name}",
                error=f"Command '{command_name}' not found"
            )

        # Execute handler
        return self._execute_handler(handler, context)

    def _find_async_handler(self, command_name: str) -> Optional[AsyncCommandHandler]:
        """
        Find async handler for normalized command name.
        
        Args:
            command_name: Normalized command name
            
        Returns:
            AsyncCommandHandler instance or None if not found
        """
        return self.async_handlers.get(command_name)
    
    async def _execute_async_handler(self, handler: AsyncCommandHandler, context: CommandContext) -> CommandResult:
        """
        Execute async handler and handle exceptions.
        
        Args:
            handler: AsyncCommandHandler instance
            context: Command context
            
        Returns:
            CommandResult from handler or error result if exception occurred
        """
        try:
            return await handler.handle(context)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении команды: {e}",
                error=str(e)
            )
    
    async def dispatch_async(self, context: CommandContext) -> CommandResult:
        """
        Dispatch a command to its async handler.

        Args:
            context: Command context with command name and arguments

        Returns:
            CommandResult from handler, or error result if command not found
        """
        # Normalize command name using shared method
        command_name = self._normalize_command_name(context.command_name)

        # Find async handler first
        async_handler = self._find_async_handler(command_name)
        if async_handler:
            return await self._execute_async_handler(async_handler, context)

        # Fallback to sync handler if no async handler found
        handler = self._find_handler(command_name)
        if handler:
            return self._execute_handler(handler, context)

        # Command not found
        return CommandResult(
            success=False,
            message=f"Неизвестная команда: /{command_name}",
            error=f"Command '{command_name}' not found"
        )

