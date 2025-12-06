# src/storage/vector_store.py

from typing import List, Optional, Dict, Any
import chromadb

from src.core.embedding import EmbeddingClient


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "default",
        max_batch_size: int = 5000,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        """
        vector store without internal chroma embedder
        expects precomputed embeddings for stored docs
        and uses the same embedding client for queries
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # embeddings always provided explicitly
        )
        # chroma limit is around 5461, keep some margin
        self.max_batch_size = max_batch_size
        self.embedding_client = embedding_client or EmbeddingClient()

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
        self.collection.delete(where={})
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
