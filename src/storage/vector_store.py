# src/storage/vector_store.py

from typing import List, Optional, Dict, Any, Union
import chromadb
from chromadb.config import Settings

from src.core.embedding import EmbeddingClient, LocalEmbeddingClient
from src.lib.syslog2 import *
import os

# Completely disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NO_SIGNAL"] = "True"


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "embed-chunks",
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
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # embeddings always provided explicitly
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity instead of L2
        )
        # chroma limit is around 5461, keep some margin
        self.max_batch_size = max_batch_size
        self.embedding_client = embedding_client or EmbeddingClient()
        
        # topics_l1_collection and topics_l2_collection removed - clustering is deprecated

    def _validate_batch_inputs(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[int, List[Optional[Dict[str, Any]]]]:
        """
        Validate batch input data and prepare metadatas.
        
        Args:
            ids: List of document IDs
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            Tuple of (total_count, prepared_metadatas)
            
        Raises:
            ValueError: If input validation fails
        """
        total = len(documents)
        if total == 0:
            return 0, []

        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, embeddings must have same length")

        if metadatas is None:
            prepared_metadatas = [None] * total
        elif len(metadatas) != total:
            raise ValueError("metadatas must match documents length")
        else:
            prepared_metadatas = metadatas

        return total, prepared_metadatas
    
    def _process_batch(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Optional[Dict[str, Any]]],
        total: int,
        show_progress: bool = True,
    ) -> None:
        """
        Process documents in batches and add to collection.
        
        Args:
            ids: List of document IDs
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries (or None)
            total: Total number of documents
            show_progress: Whether to show progress
        """
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
                print(f"\rvector_store progress: {added}/{total} ({pct}%)", flush=True, end="")

        if show_progress:
            print()
        if show_progress and total > 0:
            syslog2(LOG_DEBUG, "vector_store batch complete", total=total)

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
        total, prepared_metadatas = self._validate_batch_inputs(ids, documents, embeddings, metadatas)
        if total == 0:
            return

        self._process_batch(ids, documents, embeddings, prepared_metadatas, total, show_progress)



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

        query_embs = self._compute_query_embeddings(query_texts)
        return self._execute_vector_query(query_embs, n_results)

    def _compute_query_embeddings(self, query_texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for query texts.
        
        Args:
            query_texts: List of query text strings
            
        Returns:
            List of embedding vectors
        """
        return self.embedding_client.get_embeddings(query_texts)

    def _execute_vector_query(self, query_embeddings: List[List[float]], n_results: int) -> Dict[str, Any]:
        """
        Execute vector query using precomputed embeddings.
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results (ids, documents, distances, etc.)
        """
        return self.collection.query(
            query_embeddings=query_embeddings,
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

    def get_or_create_collection(self, collection_name: str):
        """
        Get or create a collection by name.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB Collection object
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # embeddings always provided explicitly
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity instead of L2
        )

    def get_embeddings_by_ids(self, ids: List[str], collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get embeddings by chunk IDs from the specified collection.
        
        Args:
            ids: List of chunk IDs
            collection_name: Collection name (default: main collection)
            
        Returns:
            Dictionary with 'ids' and 'embeddings' keys
        """
        collection = self.collection if collection_name is None else self.get_or_create_collection(collection_name)
        
        if not ids:
            return {"ids": [], "embeddings": []}
        
        # ChromaDB get() can handle multiple IDs
        result = collection.get(ids=ids, include=["embeddings", "metadatas"])
        
        return {
            "ids": result.get("ids", []),
            "embeddings": result.get("embeddings", []),
            "metadatas": result.get("metadatas", [])
        }

    def update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any], collection_name: Optional[str] = None) -> None:
        """
        Update metadata for a chunk in the collection.
        
        Args:
            chunk_id: ID of the chunk to update
            metadata: New metadata dictionary
            collection_name: Collection name (default: main collection)
        """
        collection = self.collection if collection_name is None else self.get_or_create_collection(collection_name)
        
        # ChromaDB update() method
        collection.update(
            ids=[chunk_id],
            metadatas=[metadata]
        )
    
    def recreate_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Recreate collection by deleting and recreating it.
        Useful when collection dimension needs to change or collection needs to be reset.
        
        Args:
            collection_name: Collection name to recreate (default: main collection)
        """
        target_collection_name = collection_name if collection_name is not None else self.collection_name
        
        # Delete old collection if it exists
        try:
            self.client.delete_collection(name=target_collection_name)
            syslog2(LOG_DEBUG, "collection deleted", name=target_collection_name)
        except Exception as e:
            syslog2(LOG_DEBUG, "error deleting collection (may not exist)", name=target_collection_name, error=str(e))
        
        # Create new collection
        new_collection = self.client.get_or_create_collection(
            name=target_collection_name,
            embedding_function=None,  # embeddings always provided explicitly
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity instead of L2
        )
        
        # Update self.collection if it's the main collection
        if collection_name is None or collection_name == self.collection_name:
            self.collection = new_collection
        
        syslog2(LOG_NOTICE, "collection recreated", name=target_collection_name)