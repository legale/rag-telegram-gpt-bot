import sys
import os

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
    def __init__(self, db_url: str = "sqlite:///legale_bot.db", vector_db_path: str = "chroma_db"):
        self.parser = ChatParser()
        self.chunker = TextChunker()
        self.db = Database(db_url)
        self.vector_store = VectorStore(persist_directory=vector_db_path)
        
    def run(self, file_path: str, clear_existing: bool = False):
        """
        Runs the ingestion pipeline.
        
        Args:
            file_path: Path to the chat dump file.
            clear_existing: Whether to clear existing data before ingestion.
        """
        print(f"Starting ingestion for {file_path}...")
        
        if clear_existing:
            print("Clearing existing data...")
            self.db.clear()
            self.vector_store.clear()
            print("Data cleared.")

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
    parser.add_argument("file", help="Path to the chat dump file (JSON or text)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    pipeline = IngestionPipeline()
    pipeline.run(args.file, clear_existing=args.clear)
