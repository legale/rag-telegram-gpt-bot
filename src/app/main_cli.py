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
    from src.core.dispatcher import CommandDispatcher
    from src.core.commands import (
        StartCommandHandler,
        HelpCommandHandler,
        ResetCommandHandler,
        TokensCommandHandler,
        ModelCommandHandler,
        FindCommandHandler,
    )
    from src.core.admin_commands import (
        AdminSetCommandHandler,
        AdminGetCommandHandler,
        AdminCommandHandler,
    )

    dispatcher = CommandDispatcher()

    # Register synchronous command handlers
    dispatcher.register("start", StartCommandHandler())
    dispatcher.register("help", HelpCommandHandler())
    dispatcher.register("reset", ResetCommandHandler(bot))
    dispatcher.register("tokens", TokensCommandHandler(bot))
    dispatcher.register("model", ModelCommandHandler(bot, admin_manager))
    dispatcher.register("find", FindCommandHandler(bot, admin_manager, debug_rag))

    # Register asynchronous admin command handlers
    if admin_manager:
        dispatcher.register_async("admin_set", AdminSetCommandHandler(admin_manager))
        dispatcher.register_async("admin_get", AdminGetCommandHandler(admin_manager))
    
    if admin_router:
        dispatcher.register_async("admin", AdminCommandHandler(admin_router))

    return dispatcher


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
    if not command.startswith("/"):
        return None

    # Parse command name and arguments
    parts = command.split(maxsplit=1)
    command_name = parts[0]
    args_text = parts[1] if len(parts) > 1 else ""

    # Create context
    context = CommandContext(
        user_id=user_id,
        chat_id=chat_id,
        command_name=command_name,
        args=args_text.split() if args_text else [],
    )

    # Dispatch command
    result = dispatcher.dispatch(context)

    # Return message (or None if command not found and not handled)
    if result.success or result.error:
        return result.message
    return None

