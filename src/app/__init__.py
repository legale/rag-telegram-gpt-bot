"""App package - composition root and dependency injection."""

from .bootstrap import create_hybrid_search, create_app
from .app import App

__all__ = ["create_hybrid_search", "create_app", "App"]

