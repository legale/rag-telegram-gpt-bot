"""Vector storage implementation using ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import os
import hashlib
import chromadb
from chromadb.config import Settings

from src.core.embedding import EmbeddingClient, LocalEmbeddingClient
from src.core.interfaces import VectorIndex, VectorDoc, ScoredDoc
from src.lib.syslog2 import *

# Completely disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NO_SIGNAL"] = "True"

# Reuse a single Settings instance process-wide to avoid Chroma shared-system
# conflicts ("already exists ... with different settings") when multiple clients
# are created during tests.
_CHROMA_SETTINGS = Settings()


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
        if not persist_directory:
            raise ValueError("persist_directory must be provided")

        persist_path = Path(persist_directory)
        if persist_path.exists() and persist_path.is_file():
            raise ValueError(f"persist_directory must be a directory, got file: {persist_directory}")
        persist_path.mkdir(parents=True, exist_ok=True)

        self.persist_directory = str(persist_path)
        # Public name (used by code/tests); internal name may be adjusted in fallback mode
        # to avoid collisions when multiple in-memory stores are created in one process.
        self.collection_name = collection_name
        self._collection_name_internal = collection_name
        self.is_persistent = True
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=_CHROMA_SETTINGS,
            )
        except Exception as e:
            # In restricted/sandboxed environments, Chroma's persistent (SQLite) backend
            # may fail to open database files. Fall back to in-memory client so core
            # functionality (and tests) can still run.
            self.is_persistent = False
            try:
                syslog2(LOG_WARNING, "chroma persistent client unavailable, falling back to in-memory", error=str(e), path=self.persist_directory)
            except Exception:
                pass
            # Important: isolate in-memory instances; otherwise different tests
            # (and different VectorStore instances) may share the same ephemeral
            # backend and collide on collection content/dimensions.
            digest = hashlib.sha1(self.persist_directory.encode("utf-8")).hexdigest()[:12]
            self._collection_name_internal = f"{collection_name}__{digest}"
            self.client = chromadb.EphemeralClient(settings=_CHROMA_SETTINGS)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name_internal,
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


class ChromaVectorIndex:
    """ChromaDB adapter implementing VectorIndex protocol."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the adapter with a VectorStore instance.

        Args:
            vector_store: VectorStore instance from src.storage.vector_store
        """
        self.vector_store = vector_store

    def upsert(self, items: List[VectorDoc]) -> None:
        """
        Upsert (insert or update) vector documents.

        Args:
            items: List of VectorDoc objects to upsert
        """
        if not items:
            return

        # Extract data from VectorDoc objects
        ids = [item.id for item in items]
        embeddings = [item.vector for item in items]
        metadatas = [item.meta for item in items]
        
        # ChromaDB expects documents (text), but we may not have them
        # Use empty strings or extract from metadata if available
        documents = []
        for item in items:
            # Try to get text from metadata, otherwise use empty string
            doc_text = item.meta.get("text", "") if isinstance(item.meta, dict) else ""
            documents.append(doc_text)

        # Use VectorStore's add_documents_with_embeddings method
        self.vector_store.add_documents_with_embeddings(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            show_progress=False  # Disable progress for programmatic use
        )

    def query(self, vector: List[float], top_k: int, filter: Optional[Dict] = None) -> List[ScoredDoc]:
        """
        Query the vector index with a pre-computed embedding vector.

        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of ScoredDoc objects
        """
        if not vector:
            return []

        # Query ChromaDB collection directly with pre-computed embedding
        # VectorStore.query expects text, so we need to use the collection directly
        collection = self.vector_store.collection
        
        # Convert filter dict to ChromaDB where clause format if provided
        where = self._convert_filter_to_where(filter)

        # Query with the embedding vector
        # Include documents to avoid reading from SQLite for each candidate
        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"]
        )

        # Convert ChromaDB results to ScoredDoc objects
        return self._convert_results_to_scored_docs(results)

    def _convert_filter_to_where(self, filter: Optional[Dict]) -> Optional[Dict]:
        """
        Convert filter dict to ChromaDB where clause format.

        Args:
            filter: Optional metadata filter dict

        Returns:
            ChromaDB where clause dict or None
        """
        if filter:
            return filter  # ChromaDB uses where clause format
        return None

    def _convert_results_to_scored_docs(self, results: Dict[str, Any]) -> List[ScoredDoc]:
        """
        Convert ChromaDB query results to ScoredDoc objects.

        Args:
            results: ChromaDB query results dict

        Returns:
            List of ScoredDoc objects with chunk_id, score, and metadata (including text)
            to avoid reading chunks from SQLite for each candidate
        """
        scored_docs = []
        
        # ChromaDB returns results in nested lists (one per query)
        if results and "ids" in results and len(results["ids"]) > 0:
            ids_list = results["ids"][0] if results["ids"] else []
            distances_list = results["distances"][0] if results.get("distances") and results["distances"] else []
            metadatas_list = results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []
            documents_list = results.get("documents", [[]])[0] if results.get("documents") and results["documents"] else []

            for i, doc_id in enumerate(ids_list):
                # ChromaDB returns distances (lower is better), convert to score (higher is better)
                # We use cosine similarity: distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance, and score = cosine_similarity
                distance = distances_list[i] if i < len(distances_list) else 1.0
                
                # For cosine similarity: distance = 1 - similarity, so similarity = 1 - distance
                # Clamp to [0, 1] range
                score = max(0.0, min(1.0, 1.0 - distance))

                # Get metadata and include text from documents to minimize SQLite roundtrips
                metadata = metadatas_list[i].copy() if i < len(metadatas_list) and metadatas_list[i] else {}
                
                # Ensure text is present in metadata to avoid SQLite roundtrips.
                # ChromaDB stores text in the "documents" field; for legacy data it can be empty.
                if "text" not in metadata:
                    document_text = documents_list[i] if i < len(documents_list) else ""
                    metadata["text"] = document_text or ""

                scored_docs.append(ScoredDoc(
                    id=doc_id,
                    score=score,
                    meta=metadata
                ))

        return scored_docs

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        # Use ChromaDB collection's delete method
        collection = self.vector_store.collection
        collection.delete(ids=ids)

    def count(self) -> int:
        """
        Get total number of documents in the index.

        Returns:
            Number of documents
        """
        return self.vector_store.count()

    def _fetch_embeddings(self, ids: List[str]) -> Dict[str, Any]:
        """
        Fetch embeddings from collection by document IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dictionary with 'ids' and 'embeddings' keys
        """
        return self.vector_store.get_embeddings_by_ids(ids)
    
    def _convert_to_dict(self, result: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Convert embeddings result to dictionary format.
        
        Args:
            result: Dictionary with 'ids' and 'embeddings' keys
            
        Returns:
            Dictionary mapping document ID to embedding vector
        """
        embeddings_dict = {}
        result_ids = result.get("ids", [])
        result_embeddings = result.get("embeddings", [])
        
        for i, doc_id in enumerate(result_ids):
            if i < len(result_embeddings):
                embeddings_dict[doc_id] = result_embeddings[i]
        
        return embeddings_dict

    def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings by document IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary mapping document ID to embedding vector
        """
        if not ids:
            return {}

        # Fetch embeddings from collection
        result = self._fetch_embeddings(ids)
        
        # Convert to dict format: {id: embedding}
        return self._convert_to_dict(result)
