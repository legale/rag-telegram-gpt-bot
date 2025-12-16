"""Use case for generating embeddings for chunks."""

from __future__ import annotations

from typing import List, Optional

from src.core.domain import Chunk
from src.core.interfaces import ChunkStore, Embedder
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

        # Get chunks without embeddings
        chunks_to_embed = self._get_chunks_without_embeddings(chunk_ids)
        
        if not chunks_to_embed:
            syslog2(LOG_NOTICE, "all specified chunks already have embeddings")
            return 0

        syslog2(LOG_NOTICE, "chunks to embed", total=len(chunks_to_embed))

        # Generate embeddings
        chunks_with_embeddings = self._generate_embeddings(chunks_to_embed, batch_size)

        # Save embeddings
        saved_count = self._save_embeddings(chunks_with_embeddings)

        syslog2(LOG_NOTICE, "embedding generation complete", processed=saved_count)
        return saved_count
    
    def _get_chunks_without_embeddings(self, chunk_ids: Optional[List[str]]) -> List[Chunk]:
        """
        Get chunks that don't have embeddings yet.
        
        Args:
            chunk_ids: Optional list of chunk IDs to check (if None, raises NotImplementedError)
            
        Returns:
            List of Chunk objects without embeddings
            
        Raises:
            NotImplementedError: If chunk_ids is None (getting all chunks without embeddings not supported)
        """
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
        return chunks_to_embed
    
    def _generate_embeddings(self, chunks: List[Chunk], batch_size: int) -> List[Chunk]:
        """
        Generate embeddings for chunks in batches.
        
        Args:
            chunks: List of Chunk objects to generate embeddings for
            batch_size: Batch size for embedding generation
            
        Returns:
            List of Chunk objects with embeddings
        """
        chunks_with_embeddings: List[Chunk] = []
        total = len(chunks)
        processed = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            batch_embeddings = self.embedder.embed_documents(batch_texts)
            
            # Update chunks with embeddings
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
                chunks_with_embeddings.append(updated_chunk)
            
            processed += len(batch)
            pct = (processed * 100) // total if total > 0 else 0
            print(f"\rProcessing embeddings: {processed}/{total} ({pct}%)", flush=True, end="")

        print()  # Newline after progress
        return chunks_with_embeddings
    
    def _save_embeddings(self, chunks_with_embeddings: List[Chunk]) -> int:
        """
        Save chunks with embeddings to store.
        
        Args:
            chunks_with_embeddings: List of Chunk objects with embeddings
            
        Returns:
            Number of chunks saved
        """
        # Save updated chunks (ChunkStore.save_batch should handle updates)
        self.chunk_store.save_batch(chunks_with_embeddings)
        return len(chunks_with_embeddings)

