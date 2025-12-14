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


class CommandDispatcher:
    """
    Dispatches commands to appropriate handlers.

    This class provides a centralized way to route commands to their handlers,
    making it easier to add new commands and test command handling logic.
    """

    def __init__(self):
        """Initialize dispatcher with empty handler registry."""
        self.handlers: Dict[str, CommandHandler] = {}

    def register(self, command_name: str, handler: CommandHandler) -> None:
        """
        Register a command handler.

        Args:
            command_name: Command name (e.g., "start", "help", "find")
            handler: CommandHandler instance
        """
        # Normalize command name (remove leading slash, lowercase)
        normalized = command_name.lstrip("/").lower()
        self.handlers[normalized] = handler

    def dispatch(self, context: CommandContext) -> CommandResult:
        """
        Dispatch a command to its handler.

        Args:
            context: Command context with command name and arguments

        Returns:
            CommandResult from handler, or error result if command not found
        """
        # Normalize command name
        command_name = context.command_name.lstrip("/").lower()

        # Find handler
        handler = self.handlers.get(command_name)

        if not handler:
            return CommandResult(
                success=False,
                message=f"Неизвестная команда: /{command_name}",
                error=f"Command '{command_name}' not found"
            )

        try:
            # Execute handler
            return handler.handle(context)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка при выполнении команды: {e}",
                error=str(e)
            )

