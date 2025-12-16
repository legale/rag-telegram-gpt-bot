"""App package - composition root and dependency injection."""

from .bootstrap import create_hybrid_search, create_app

def __getattr__(name: str):
    if name == "App":
        from .app import App

        return App
    raise AttributeError(name)


__all__ = ["create_hybrid_search", "create_app", "App"]
