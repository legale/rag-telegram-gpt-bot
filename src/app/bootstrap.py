"""Bootstrap module for dependency injection and application composition."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.storage.db import Database
from src.core.embedding import EmbeddingClient, LocalEmbeddingClient, create_embedding_client

from src.storage.sqlite import SqliteMessageStore, SqliteChunkStore, SqliteFTSIndex
from src.storage.vector import VectorStore, ChromaVectorIndex
# Adapters removed: EmbedderAdapter, LLMAdapter - clients implement protocols directly
from src.core.search import HybridSearch
from src.core.hybrid_retrieval import HybridRetrievalService
# RetrievalService removed - legacy RAG code
from src.core.llm import LLMClient
from src.lib.syslog2 import *


def create_embedding_client_from_config(
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient],
    profile_dir: Optional[str | Path]
) -> EmbeddingClient | LocalEmbeddingClient:
    """
    Create embedding client from config or use provided one.
    
    This function handles the choice between local and API embedding clients
    based on configuration. Core modules should use only the Embedder interface
    and not make this choice themselves.
    
    Args:
        embedding_client: Optional pre-configured embedding client
        profile_dir: Optional profile directory for loading embedding config
        
    Returns:
        EmbeddingClient or LocalEmbeddingClient instance
    """
    if embedding_client is not None:
        return embedding_client
    
    # Try to load from profile config if available
    if profile_dir:
        profile_path = Path(profile_dir)
        if profile_path.exists():
            try:
                from src.bot.config import BotConfig
                config = BotConfig(profile_path)
                generator = config.embedding_generator
                model = config.embedding_model
                
                # Choose between local and API based on generator
                # This logic is moved here from core/embedding.py
                generator_lower = generator.lower() if generator else "openrouter"
                
                if generator_lower == "local":
                    # Create local embedding client
                    from src.core.embedding import LocalEmbeddingClient
                    local_model = model or "paraphrase-multilingual-mpnet-base-v2"
                    return LocalEmbeddingClient(model=local_model)
                else:
                    # Create API embedding client (openrouter, openai, etc.)
                    api_model = model or "text-embedding-3-small"
                    return EmbeddingClient(model=api_model)
            except Exception:
                # Fall back to default if config load fails
                return EmbeddingClient()
    
    # Default fallback to API client
    return EmbeddingClient()


def create_hybrid_search(
    db_url: str,
    vector_db_path: str,
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient] = None,
    profile_dir: Optional[str | Path] = None,
) -> HybridSearch:
    """
    Create and configure HybridSearch use case with all dependencies.

    This function serves as the composition root, creating all adapters and
    wiring them together to create a fully configured HybridSearch instance.

    Args:
        db_url: SQLite database URL (e.g., "sqlite:///path/to/db")
        vector_db_path: Path to ChromaDB persistent directory
        embedding_client: Optional pre-configured embedding client.
                         If None, creates default EmbeddingClient.
        profile_dir: Optional profile directory for loading embedding config

    Returns:
        Configured HybridSearch use case instance
    """
    # Create or use provided embedding client
    embedding_client = create_embedding_client_from_config(embedding_client, profile_dir)

    # Create infrastructure instances
    database = Database(db_url)
    vector_store = VectorStore(
        persist_directory=vector_db_path,
        embedding_client=embedding_client
    )

    # Create adapters
    message_store = SqliteMessageStore(database)
    chunk_store = SqliteChunkStore(database)
    vector_index = ChromaVectorIndex(vector_store)
    # embedder = EmbedderAdapter(embedding_client) -> Removed

    # Create and return use case
    hybrid_search = HybridSearch(
        embedder=embedding_client, # Passed directly
        vector_index=vector_index,
        chunk_store=chunk_store,
        message_store=message_store,
    )

    return hybrid_search


# create_retrieval_service removed - legacy RAG code

def create_hybrid_retrieval(
    db_url: str,
    vector_db_path: str,
    embedding_client: Optional[EmbeddingClient | LocalEmbeddingClient] = None,
    profile_dir: Optional[str | Path] = None,
    log_level: int = LOG_WARNING,
    fts_only: bool = False,
    llm_client: Optional[LLMClient] = None,
    retrieval_mode: str = "hybrid",
) -> HybridRetrievalService:
    """
    Create and configure HybridRetrievalService with all dependencies.

    This function creates FTS5 + vector hybrid retrieval service.

    Args:
        db_url: SQLite database URL (e.g., "sqlite:///path/to/db")
        vector_db_path: Path to ChromaDB persistent directory
        embedding_client: Optional pre-configured embedding client.
                         If None, creates default EmbeddingClient.
        profile_dir: Optional profile directory for loading embedding config
        log_level: Logging level (LOG_WARNING=4 by default)
        fts_only: If True, skip vector reranking and use FTS-only mode (deprecated, use retrieval_mode)
        llm_client: Optional LLM client for query rephrasing before vector search
        retrieval_mode: "fts_only", "hybrid", or "vector_only"

    Returns:
        Configured HybridRetrievalService instance
    """
    # Create or use provided embedding client
    embedding_client = create_embedding_client_from_config(embedding_client, profile_dir)

    # Create infrastructure instances
    database = Database(db_url)
    vector_store = VectorStore(
        persist_directory=vector_db_path,
        embedding_client=embedding_client
    )

    # Create adapters
    message_store = SqliteMessageStore(database)
    chunk_store = SqliteChunkStore(database)
    vector_index = ChromaVectorIndex(vector_store)
    # embedder = EmbedderAdapter(embedding_client) -> Removed, using client directly
    fts_index = SqliteFTSIndex(database)
    
    # Create LLM adapter if llm_client is provided (for query rephrasing)
    # llm = LLMAdapter(llm_client) if llm_client else None -> Removed, using client directly
    llm = llm_client

    # Create and return HybridRetrievalService
    # Map retrieval_mode to fts_only if needed
    if retrieval_mode == "fts_only":
        fts_only = True
    elif retrieval_mode == "vector_only":
        # For vector_only, we still need fts_only=False but will handle in service
        fts_only = False
    # else: hybrid mode, fts_only already set correctly
    
    hybrid_retrieval = HybridRetrievalService(
        fts_index=fts_index,
        vector_index=vector_index,
        embedder=embedding_client, # Passed directly
        chunk_store=chunk_store,
        message_store=message_store,
        log_level=log_level,
        fts_only=fts_only,
        llm=llm,
    )

    return hybrid_retrieval


def create_app(
    db_url: str,
    vector_db_path: str,
    model_name: Optional[str] = None,
    log_level: int = LOG_WARNING,
    debug_rag: bool = False,
    profile_dir: Optional[str | Path] = None,
    retrieval_type: str = "hybrid",
) -> 'App':
    """
    Create and configure App with all dependencies.
    
    This function creates LegaleBot, AdminManager, and App instance.
    
    Args:
        db_url: Database URL
        vector_db_path: Vector database path
        model_name: Model name
        log_level: Logging level
        debug_rag: Whether to enable debug RAG mode
        profile_dir: Optional profile directory path
        retrieval_type: Retrieval type
        
    Returns:
        Configured App instance
    """
    from types import MethodType

    from src.app.app import App
    from src.app.main_cli import register_async_handlers, register_sync_handlers
    from src.bot.admin import AdminManager
    from src.bot.admin_router import AdminCommandRouter
    from src.bot.core import LegaleBot
    from src.core.command_service import CommandService
    
    # Create bot
    bot = LegaleBot(
        db_url=db_url,
        vector_db_path=vector_db_path,
        model_name=model_name,
        log_level=log_level,
        debug_rag=debug_rag,
        profile_dir=profile_dir,
        retrieval_type=retrieval_type
    )
    
    # Create AdminManager if profile_dir is available
    admin_manager = None
    if profile_dir:
        try:
            profile_path = Path(profile_dir)
            admin_manager = AdminManager(profile_path)
        except Exception:
            # Continue without admin_manager if it fails
            pass

    admin_router = None
    if admin_manager is not None:
        try:
            admin_router = AdminCommandRouter(admin_manager)
        except Exception:
            admin_router = None

    command_service = CommandService()
    register_sync_handlers(command_service, bot, admin_manager=admin_manager, debug_rag=debug_rag)
    if admin_manager is not None or admin_router is not None:
        register_async_handlers(command_service, admin_manager=admin_manager, admin_router=admin_router)
    
    # Create App with unified entry points
    app = App(
        bot=bot,
        command_service=command_service,
        admin_manager=admin_manager,
        debug_rag=debug_rag
    )

    def _ingest(*_args, **_kwargs):
        raise NotImplementedError("App.ingest is not implemented yet")

    app.ingest = MethodType(_ingest, app)
    
    return app
