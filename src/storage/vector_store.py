import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

class VectorStore:
    """Wrapper around ChromaDB for storing and retrieving vector embeddings."""
    
    def __init__(self, persist_directory: str, collection_name: str = "chat_chunks"):
        if not persist_directory:
            raise ValueError("persist_directory must be provided")
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def clear(self) -> int:
        """Deletes and recreates the collection, returning the number of removed documents."""
        removed = self.count()
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return removed
        
    def add_documents(self, ids: List[str], documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Adds documents to the vector store.
        
        Args:
            ids: List of unique identifiers for the documents.
            documents: List of text content to be embedded and stored.
            metadatas: Optional list of metadata dictionaries for each document.
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Queries the vector store for similar documents.
        
        Args:
            query_text: The query text.
            n_results: Number of results to return.
            
        Returns:
            Dictionary containing query results (ids, distances, metadatas, documents).
        """
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
    def count(self) -> int:
        """Returns the number of documents in the collection."""
        return self.collection.count()
