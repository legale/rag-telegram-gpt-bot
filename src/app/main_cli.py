"""Main CLI entry point using CommandDispatcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from src.core.dispatcher import CommandDispatcher, CommandContext
from src.core.commands import (
    StartCommandHandler,
    HelpCommandHandler,
    ResetCommandHandler,
    TokensCommandHandler,
    ModelCommandHandler,
    FindCommandHandler,
)
from src.bot.core import LegaleBot
from src.bot.admin import AdminManager
from src.bot.admin_router import AdminCommandRouter
from src.lib.syslog2 import *


def _register_sync_handlers(
    dispatcher: CommandDispatcher,
    bot: LegaleBot,
    admin_manager: Optional[AdminManager] = None,
    debug_rag: bool = False
) -> None:
    """
    Register synchronous command handlers.
    
    Args:
        dispatcher: CommandDispatcher instance
        bot: LegaleBot instance
        admin_manager: Optional AdminManager instance
        debug_rag: Whether to enable debug RAG mode
    """
    dispatcher.register("start", StartCommandHandler())
    dispatcher.register("help", HelpCommandHandler())
    dispatcher.register("reset", ResetCommandHandler(bot))
    dispatcher.register("tokens", TokensCommandHandler(bot))
    dispatcher.register("model", ModelCommandHandler(bot, admin_manager))
    dispatcher.register("find", FindCommandHandler(bot, admin_manager, debug_rag))

def _register_async_handlers(
    dispatcher: CommandDispatcher,
    admin_manager: Optional[AdminManager] = None,
    admin_router: Optional[AdminCommandRouter] = None
) -> None:
    """
    Register asynchronous admin command handlers.
    
    Args:
        dispatcher: CommandDispatcher instance
        admin_manager: Optional AdminManager instance
        admin_router: Optional AdminCommandRouter instance
    """
    from src.core.admin_commands import (
        AdminSetCommandHandler,
        AdminGetCommandHandler,
        AdminCommandHandler,
    )
    
    if admin_manager:
        dispatcher.register_async("admin_set", AdminSetCommandHandler(admin_manager))
        dispatcher.register_async("admin_get", AdminGetCommandHandler(admin_manager))
    
    if admin_router:
        dispatcher.register_async("admin", AdminCommandHandler(admin_router))

def create_dispatcher(
    bot: LegaleBot,
    admin_manager: Optional[AdminManager] = None,
    admin_router: Optional[AdminCommandRouter] = None,
    debug_rag: bool = False
) -> CommandDispatcher:
    """
    Create and configure CommandDispatcher with all command handlers.

    Args:
        bot: LegaleBot instance
        admin_manager: Optional AdminManager instance
        admin_router: Optional AdminCommandRouter instance
        debug_rag: Whether to enable debug RAG mode

    Returns:
        Configured CommandDispatcher instance
    """
    dispatcher = CommandDispatcher()

    # Register synchronous command handlers
    _register_sync_handlers(dispatcher, bot, admin_manager, debug_rag)

    # Register asynchronous admin command handlers
    _register_async_handlers(dispatcher, admin_manager, admin_router)

    return dispatcher


def parse_command(text: str) -> tuple[Optional[str], str]:
    """
    Parse command text into command name and arguments.
    
    Args:
        text: Command text (e.g., "/find 2.0 vpn туннель" or "/help")
        
    Returns:
        Tuple of (command_name, args_text) or (None, text) if not a command
    """
    text = text.strip()
    if not text.startswith("/"):
        return None, text
    
    parts = text.split(maxsplit=1)
    command_name = parts[0]
    args_text = parts[1] if len(parts) > 1 else ""
    return command_name, args_text


def handle_command(
    command: str,
    dispatcher: CommandDispatcher,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None
) -> Optional[str]:
    """
    Handle a command string using dispatcher.

    Args:
        command: Command string (e.g., "/find 2.0 vpn туннель" or "/help")
        dispatcher: CommandDispatcher instance
        user_id: Optional user ID
        chat_id: Optional chat ID

    Returns:
        Response string if command was handled, None if not a command
    """
    # Parse command name and arguments
    command_name, args = _parse_command(command)
    if not command_name:
        return None

    # Create context
    context = _create_command_context(command_name, args, user_id, chat_id)

    # Dispatch command
    result = dispatcher.dispatch(context)

    # Return message (or None if command not found and not handled)
    if result.success or result.error:
        return result.message
    return None


def _parse_command(command: str) -> tuple[Optional[str], list[str]]:
    """
    Parse command string into command name and arguments list.
    
    Args:
        command: Command string (e.g., "/find 2.0 vpn туннель" or "/help")
        
    Returns:
        Tuple of (command_name, args_list) or (None, []) if not a command
    """
    command_name, args_text = parse_command(command)
    if not command_name:
        return None, []
    
    args = args_text.split() if args_text else []
    return command_name, args


def _create_command_context(
    command_name: str,
    args: list[str],
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None
) -> CommandContext:
    """
    Create CommandContext from parsed command data.
    
    Args:
        command_name: Command name (e.g., "/find", "/help")
        args: List of command arguments
        user_id: Optional user ID
        chat_id: Optional chat ID
        
    Returns:
        CommandContext instance
    """
    return CommandContext(
        user_id=user_id,
        chat_id=chat_id,
        command_name=command_name,
        args=args,
    )


async def handle_command_async(
    command: str,
    dispatcher: CommandDispatcher,
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    metadata: Optional[dict] = None
) -> tuple[Optional[str], Optional[dict]]:
    """
    Handle a command string asynchronously using dispatcher.

    Args:
        command: Command string (e.g., "/admin list" or "/find query")
        dispatcher: CommandDispatcher instance
        user_id: Optional user ID
        chat_id: Optional chat ID
        metadata: Optional metadata dict to pass to context

    Returns:
        Tuple of (response_message, result_data) where result_data contains
        special handling flags like needs_formatting. Returns (None, None) if not a command.
    """
    # Parse command name and arguments
    command_name, args_text = parse_command(command)
    if not command_name:
        return None, None

    # Normalize /set_admin to /admin_set
    if command_name == "/set_admin":
        command_name = "/admin_set"

    # Create context
    context = CommandContext(
        user_id=user_id,
        chat_id=chat_id,
        command_name=command_name,
        args=args_text.split() if args_text else [],
        metadata=metadata or {},
    )

    # Check if this is an admin command (async handler)
    admin_commands = ["/admin", "/admin_set", "/admin_get"]
    is_admin_command = command_name.lower() in [c.lower() for c in admin_commands]

    if is_admin_command:
        # Use async dispatcher for admin commands
        result = await dispatcher.dispatch_async(context)
    else:
        # Use sync dispatcher for regular commands
        result = dispatcher.dispatch(context)

    # Return message and data if command was handled
    if result.success or result.error:
        return result.message, result.data
    return None, None

