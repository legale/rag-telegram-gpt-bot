import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

class VectorStore:
    """Wrapper around ChromaDB for storing and retrieving vector embeddings."""
    
    def __init__(self, collection_name: str = "chat_chunks", persist_directory: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def clear(self):
        """Deletes and recreates the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
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
