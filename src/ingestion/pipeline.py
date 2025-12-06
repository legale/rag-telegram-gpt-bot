import sys
import os
from typing import Optional

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.parser import ChatParser
from src.ingestion.chunker import TextChunker
from src.storage.db import Database, ChunkModel
from src.storage.vector_store import VectorStore
import uuid
import json

class IngestionPipeline:
    def __init__(self, db_url: str, vector_db_path: str):
        self.parser = ChatParser()
        self.chunker = TextChunker()
        self.db = Database(db_url)
        self.vector_store = VectorStore(persist_directory=vector_db_path)
    
    def _clear_data(self):
        """Clears SQL and vector storage with verbose output."""
        print("Clearing existing data...")
        db_before = self.db.count_chunks()
        print(f"SQL database ({self.db.db_url}): {db_before} chunks before cleanup.")
        removed_db = self.db.clear()
        db_after = self.db.count_chunks()
        print(f"Removed {removed_db} chunks from SQL database. Remaining: {db_after}.")
        
        vector_before = self.vector_store.count()
        print(
            f"Vector store ({self.vector_store.persist_directory}, collection '{self.vector_store.collection_name}'): "
            f"{vector_before} embeddings before cleanup."
        )
        removed_vectors = self.vector_store.clear()
        vector_after = self.vector_store.count()
        print(f"Removed {removed_vectors} embeddings from vector store. Remaining: {vector_after}.")
        print("Data cleared.")
        
    def run(self, file_path: Optional[str] = None, clear_existing: bool = False):
        """
        Runs the ingestion pipeline.
        
        Args:
            file_path: Path to the chat dump file.
            clear_existing: Whether to clear existing data before ingestion.
        """
        if clear_existing:
            self._clear_data()
            if not file_path:
                return
        
        if not file_path:
            raise ValueError("file_path must be provided when not running a cleanup-only command.")
        
        print(f"Starting ingestion for {file_path}...")

        # 1. Parse
        messages = self.parser.parse_file(file_path)
        print(f"Parsed {len(messages)} messages.")
        
        # 2. Chunk
        chunks = self.chunker.chunk_messages(messages)
        print(f"Created {len(chunks)} chunks.")
        
        # 3. Store in SQL DB
        chunk_models = []
        ids = []
        documents = []
        metadatas = []
        
        session = self.db.get_session()
        try:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                
                # SQL Model
                model = ChunkModel(
                    id=chunk_id,
                    text=chunk.text,
                    metadata_json=json.dumps(chunk.metadata)
                )
                chunk_models.append(model)
                
                # Vector Store Data
                ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(chunk.metadata)
            
            session.add_all(chunk_models)
            session.commit()
            print("Saved chunks to SQL database.")
        except Exception as e:
            session.rollback()
            print(f"Error saving to DB: {e}")
            raise
        finally:
            session.close()
            
        # 4. Store in Vector DB
        if ids:
            self.vector_store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
            print("Saved embeddings to Vector database.")
            
        print("Ingestion complete.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest chat dump into database and vector store.")
    parser.add_argument("file", nargs="?", help="Path to the chat dump file (JSON or text)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not args.file and not args.clear:
        parser.error("Please provide a chat dump file or specify --clear for cleanup.")
    
    pipeline = IngestionPipeline()
    pipeline.run(args.file, clear_existing=args.clear)
