"""Use case for generating embeddings for chunks."""

from __future__ import annotations

from typing import List, Optional

from ...domain import Chunk
from ...interfaces import ChunkStore, Embedder
from src.lib.syslog2 import *


class GenerateEmbeddings:
    """
    Use case for generating embeddings for chunks that don't have them.
    
    This use case depends only on ChunkStore and Embedder interfaces.
    Note: This is a simplified version. Full implementation would need
    ChunkStore to support querying chunks without embeddings.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        embedder: Embedder,
    ):
        """
        Initialize use case.

        Args:
            chunk_store: ChunkStore interface implementation
            embedder: Embedder interface implementation
        """
        self.chunk_store = chunk_store
        self.embedder = embedder

    def execute(self, chunk_ids: Optional[List[str]] = None, batch_size: int = 128) -> int:
        """
        Generate embeddings for chunks.

        Args:
            chunk_ids: Optional list of chunk IDs to process (if None, processes all chunks without embeddings)
            batch_size: Batch size for embedding generation

        Returns:
            Number of chunks with embeddings generated
        """
        syslog2(LOG_NOTICE, "starting embedding generation", batch_size=batch_size)

        # Note: This is a simplified implementation.
        # Full implementation would need ChunkStore to support:
        # - get_chunks_without_embeddings() method
        # - Or iterate through all chunks and filter
        
        # For now, this use case demonstrates the structure.
        # Actual implementation would require extending ChunkStore interface
        # or using a different approach (e.g., direct database access for this specific case).
        
        if chunk_ids is None:
            raise NotImplementedError(
                "Getting all chunks without embeddings requires extending ChunkStore interface. "
                "For now, provide explicit chunk_ids."
            )

        # Get chunks by IDs
        chunks = self.chunk_store.get_by_ids(chunk_ids)
        
        # Filter chunks without embeddings
        chunks_to_embed = [chunk for chunk in chunks if chunk.embedding is None]
        
        if not chunks_to_embed:
            syslog2(LOG_NOTICE, "all specified chunks already have embeddings")
            return 0

        syslog2(LOG_NOTICE, "chunks to embed", total=len(chunks_to_embed))

        # Process in batches
        processed = 0
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            batch_embeddings = self.embedder.embed_documents(batch_texts)
            
            # Update chunks with embeddings
            updated_chunks = []
            for chunk, embedding in zip(batch, batch_embeddings):
                # Create updated chunk with embedding
                updated_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    msg_ids=chunk.msg_ids,
                    valid_period=chunk.valid_period,
                    embedding=embedding,
                    metadata=chunk.metadata
                )
                updated_chunks.append(updated_chunk)
            
            # Save updated chunks (ChunkStore.save_batch should handle updates)
            self.chunk_store.save_batch(updated_chunks)
            processed += len(batch)
            
            pct = (processed * 100) // len(chunks_to_embed) if chunks_to_embed else 0
            print(f"\rProcessing embeddings: {processed}/{len(chunks_to_embed)} ({pct}%)", flush=True, end="")

        print()  # Newline after progress
        syslog2(LOG_NOTICE, "embedding generation complete", processed=processed)
        return processed

