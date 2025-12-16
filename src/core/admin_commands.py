"""Deprecated admin command handler module.

This module remains for backward-compatible imports.
Use `src.core.commands.admin` instead.
"""

from __future__ import annotations

from src.core.commands.admin import AdminCommandHandler, AdminGetCommandHandler, AdminSetCommandHandler

__all__ = ["AdminSetCommandHandler", "AdminGetCommandHandler", "AdminCommandHandler"]

