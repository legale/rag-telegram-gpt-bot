# src/storage/vector_store.py

from typing import List, Optional, Dict, Any, Union
import chromadb
from chromadb.config import Settings

from src.core.embedding import EmbeddingClient, LocalEmbeddingClient
from src.core.syslog2 import *
import os

# Completely disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NO_SIGNAL"] = "True"


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "default",
        max_batch_size: int = 5000,
        embedding_client: Optional[Union[EmbeddingClient, LocalEmbeddingClient]] = None,
    ):
        """
        vector store without internal chroma embedder
        expects precomputed embeddings for stored docs
        and uses the same embedding client for queries
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        # chroma limit is around 5461, keep some margin
        self.max_batch_size = max_batch_size
        self.embedding_client = embedding_client or EmbeddingClient()
        
        # Get expected dimension from embedding client
        self.expected_dimension = self.embedding_client.get_dimension()
        
        # Get or create collection, checking dimension compatibility
        self.collection = self._get_or_create_collection_with_dimension()
    
    def _get_or_create_collection_with_dimension(self):
        """
        Get or create collection, ensuring it matches the expected embedding dimension.
        If collection exists with wrong dimension, recreate it.
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            
            # Check if collection is empty (no dimension constraint yet)
            count = collection.count()
            if count == 0:
                # Empty collection, safe to use
                return collection
            
            # Collection has data, check dimension by sampling an embedding
            # Get a sample to check dimension
            sample = collection.get(limit=1, include=["embeddings"])
            if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                existing_dim = len(sample["embeddings"][0])
                if existing_dim != self.expected_dimension:
                    # Dimension mismatch - need to recreate collection
                    syslog2(LOG_WARNING, 
                        f"Collection dimension mismatch: expected {self.expected_dimension}, found {existing_dim}. "
                        f"Recreating collection '{self.collection_name}'...")
                    # Delete old collection
                    self.client.delete_collection(name=self.collection_name)
                    # Create new collection
                    collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=None,
                    )
                    syslog2(LOG_INFO, f"Collection '{self.collection_name}' recreated with dimension {self.expected_dimension}")
                else:
                    # Dimension matches, use existing collection
                    syslog2(LOG_DEBUG, f"Using existing collection '{self.collection_name}' with dimension {existing_dim}")
            else:
                # No embeddings found, safe to use
                return collection
            
            return collection
        except Exception as e:
            # Collection doesn't exist, try to create it
            # But handle case where collection might have been created between get and create
            try:
                collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                )
                syslog2(LOG_DEBUG, f"Created new collection '{self.collection_name}' with dimension {self.expected_dimension}")
                return collection
            except Exception as create_error:
                # Collection might have been created by another process/thread, try to get it again
                if "already exists" in str(create_error).lower():
                    try:
                        collection = self.client.get_collection(name=self.collection_name)
                        syslog2(LOG_DEBUG, f"Collection '{self.collection_name}' already exists, using existing one")
                        return collection
                    except Exception:
                        # Re-raise original error if we still can't get it
                        raise create_error
                else:
                    # Re-raise if it's a different error
                    raise create_error
    
    def _recreate_collection_with_dimension(self, new_dimension: int):
        """Recreate collection with new dimension."""
        try:
            # Delete old collection
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # Collection might not exist, ignore
            pass
        
        # Create new collection
        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,
        )
        return collection

    def add_documents_with_embeddings(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        add documents using precomputed embeddings
        ids, documents, embeddings must be aligned
        """
        total = len(documents)
        if total == 0:
            return

        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, embeddings must have same length")

        # Validate embedding dimensions and recreate collection if needed
        if embeddings:
            first_dim = len(embeddings[0])
            
            # Check all embeddings have same dimension first
            for i, emb in enumerate(embeddings):
                if len(emb) != first_dim:
                    raise ValueError(f"Embedding at index {i} has dimension {len(emb)}, expected {first_dim}")
            
            # CRITICAL: Check existing collection dimension BEFORE attempting to add
            # ChromaDB will fail if dimension doesn't match, so we must recreate collection first
            needs_recreate = False
            try:
                count = self.collection.count()
                if count > 0:
                    # Sample existing embeddings to check dimension
                    sample = self.collection.get(limit=1, include=["embeddings"])
                    if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                        existing_dim = len(sample["embeddings"][0])
                        if existing_dim != first_dim:
                            # Existing data has different dimension - MUST recreate collection
                            needs_recreate = True
                            syslog2(LOG_WARNING, 
                                f"Existing collection has dimension {existing_dim}, new embeddings have {first_dim}. "
                                f"Recreating collection '{self.collection_name}'...")
            except Exception as e:
                # If we can't check, assume collection might be empty or corrupted
                syslog2(LOG_DEBUG, f"Could not check collection dimension: {e}")
            
            # Recreate collection if needed
            if needs_recreate:
                self.collection = self._recreate_collection_with_dimension(first_dim)
                self.expected_dimension = first_dim
                syslog2(LOG_INFO, f"Collection '{self.collection_name}' recreated with dimension {first_dim}")
            
            # Update expected dimension to match actual embeddings
            if first_dim != self.expected_dimension:
                syslog2(LOG_INFO, 
                    f"Updating expected dimension: {self.expected_dimension} -> {first_dim}")
                self.expected_dimension = first_dim

        if metadatas is None:
            metadatas = [None] * total
        elif len(metadatas) != total:
            raise ValueError("metadatas must match documents length")

        batch_size = self.max_batch_size
        added = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_ids = ids[start:end]
            batch_docs = documents[start:end]
            batch_meta = metadatas[start:end]
            batch_embs = embeddings[start:end]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embs,
            )

            added = end
            if show_progress:
                pct = added * 100 // total
                print(f"\rvector_store: added {added}/{total} documents ({pct}%)", end="", flush=True)

        if show_progress and total > 0:
            print()



    def count(self) -> int:
        """how many objects in collection"""
        return self.collection.count()

    def clear(self) -> int:
        """clear collection, return how many were removed"""
        before = self.collection.count()
        if before > 0:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data and all_data.get("ids"):
                self.collection.delete(ids=all_data["ids"])
        after = self.collection.count()
        return before - after

    def query(self, query_texts: List[str], n_results: int = 3) -> Dict[str, Any]:
        """
        compute query embeddings with the same embedding client
        and pass them to chroma as query_embeddings
        """
        if not query_texts:
            return {"ids": [], "documents": [], "distances": []}

        query_embs = self.embedding_client.get_embeddings(query_texts)
        return self.collection.query(
            query_embeddings=query_embs,
            n_results=n_results,
        )

    def get_all_embeddings(self) -> Dict[str, Any]:
        """
        Returns all embeddings and ids from the collection.
        This extracts the raw vectors for clustering.
        """
        # ChromaDB get() can return headings, ids, etc.
        # We need "embeddings" which might not be returned by default.
        return self.collection.get(include=["embeddings", "metadatas", "documents"])
