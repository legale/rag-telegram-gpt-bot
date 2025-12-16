"""Orchestrator use case that coordinates ingestion pipeline stages."""

from __future__ import annotations

from typing import Optional, Dict, Any

from .ingest_messages import IngestMessages
from .process_chunks import ProcessChunks
from .generate_embeddings import GenerateEmbeddings
from .sync_to_vector_store import SyncToVectorStore
from src.lib.syslog2 import *


class PipelineOrchestrator:
    """
    Orchestrator that coordinates multiple ingestion use cases.
    
    This replaces the monolithic IngestionPipeline.run_all() method
    by composing simpler use cases.
    """

    def __init__(
        self,
        ingest_messages: IngestMessages,
        process_chunks: ProcessChunks,
        generate_embeddings: GenerateEmbeddings,
        sync_to_vector_store: SyncToVectorStore,
    ):
        """
        Initialize orchestrator with use cases.

        Args:
            ingest_messages: IngestMessages use case
            process_chunks: ProcessChunks use case
            generate_embeddings: GenerateEmbeddings use case
            sync_to_vector_store: SyncToVectorStore use case
        """
        self.ingest_messages = ingest_messages
        self.process_chunks = process_chunks
        self.generate_embeddings = generate_embeddings
        self.sync_to_vector_store = sync_to_vector_store

    def _execute_stage(
        self,
        stage_name: str,
        stage_num: int,
        stage_func: callable,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute a single pipeline stage with error handling.
        
        Args:
            stage_name: Name of the stage (e.g., "parse and store messages")
            stage_num: Stage number for logging
            stage_func: Function to execute for this stage
            *args: Positional arguments for stage_func
            **kwargs: Keyword arguments for stage_func
            
        Returns:
            Result from stage_func execution, or None if stage is not implemented
        """
        try:
            syslog2(LOG_NOTICE, f"running stage{stage_num}: {stage_name}")
            return stage_func(*args, **kwargs)
        except Exception as e:
            return self._handle_stage_error(stage_name, stage_num, e)
    
    def _handle_stage_error(self, stage_name: str, stage_num: int, error: Exception) -> None:
        """
        Handle error that occurred during stage execution.
        
        Args:
            stage_name: Name of the stage that failed
            stage_num: Stage number that failed
            error: Exception that occurred
            
        Returns:
            None (always raises exception after logging)
        """
        syslog2(LOG_ERR, f"stage{stage_num} failed", stage=stage_name, error=str(error))
        raise error

    def run_all(
        self,
        file_path: str,
        model: Optional[str] = None,
        batch_size: int = 128,
        **clustering_params
    ) -> Dict[str, Any]:
        """
        Run all ingestion stages in sequence.

        Args:
            file_path: Path to chat dump file
            model: Optional embedding model override
            batch_size: Batch size for embedding generation
            **clustering_params: Clustering parameters (for future stages)

        Returns:
            Dictionary with statistics about the ingestion process
        """
        stats = {}

        # Stage 0: Parse and store messages
        messages_saved = self._execute_stage(
            "parse and store messages",
            0,
            self.ingest_messages.execute,
            file_path
        )
        if messages_saved is not None:
            stats["stage0_messages_saved"] = messages_saved

        # Stage 1: Create and store chunks
        chunks_saved = self._execute_stage(
            "create and store chunks",
            1,
            self.process_chunks.execute
        )
        if chunks_saved is not None:
            stats["stage1_chunks_saved"] = chunks_saved

        # Stage 2: Generate embeddings
        # Note: This requires getting chunk IDs first, which needs ChunkStore extension
        # For now, this is a placeholder showing the structure
        # embeddings_generated = self._execute_stage(
        #     "generate embeddings for chunks (save to SQLite)",
        #     2,
        #     self.generate_embeddings.execute,
        #     ...
        # )
        # if embeddings_generated is not None:
        #     stats["stage2_embeddings_generated"] = embeddings_generated

        # Stage 3: Sync chunks to vector database
        # Note: This also requires getting chunk IDs first
        # chunks_synced = self._execute_stage(
        #     "sync chunks to vector database",
        #     3,
        #     self.sync_to_vector_store.execute,
        #     ...
        # )
        # if chunks_synced is not None:
        #     stats["stage3_chunks_synced"] = chunks_synced

        # Stages 4-9 (clustering and topic naming) are not yet migrated to use cases
        # They remain in IngestionPipeline for now

        syslog2(LOG_NOTICE, "pipeline orchestrator complete", **stats)
        return stats

