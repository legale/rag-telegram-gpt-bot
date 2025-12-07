# src/ingestion/pipeline.py

import sys
import os
from typing import Optional

# add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.parser import ChatParser
from src.ingestion.chunker import TextChunker
from src.storage.db import Database, ChunkModel
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, create_embedding_client
from pathlib import Path
import uuid
import json


class IngestionPipeline:
    def __init__(self, db_url: str, vector_db_path: str, collection_name: str = "default", profile_dir: Optional[str] = None):
        self.parser = ChatParser()
        self.chunker = TextChunker()
        self.db = Database(db_url)
        self.profile_dir = Path(profile_dir) if profile_dir else None
        
        # Load profile config if available
        embedding_client = None
        if self.profile_dir and self.profile_dir.exists():
            try:
                from src.bot.config import BotConfig
                config = BotConfig(self.profile_dir)
                embedding_client = create_embedding_client(
                    generator=config.embedding_generator,
                    model=config.embedding_model
                )
            except Exception as e:
                print(f"Warning: Could not load profile config: {e}")
                print("  Using default embedding client")
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name,
            embedding_client=embedding_client,
        )
        self.db_url = db_url
        self.embedding_client = embedding_client

    def _clear_data(self):
        """clears sql and vector storage with verbose output"""
        print("Clearing existing data...")
        db_before = self.db.count_chunks()
        print(f"SQL database ({self.db_url}): {db_before} chunks before cleanup.")
        removed_db = self.db.clear()
        db_after = self.db.count_chunks()
        print(f"Removed {removed_db} chunks from SQL database. Remaining: {db_after}.")

        vector_before = self.vector_store.count()
        print(
            f"Vector store ({self.vector_store.persist_directory}, "
            f"collection '{self.vector_store.collection_name}'): "
            f"{vector_before} embeddings before cleanup."
        )
        removed_vectors = self.vector_store.clear()
        vector_after = self.vector_store.count()
        print(f"Removed {removed_vectors} embeddings from vector store. Remaining: {vector_after}.")
        print("Data cleared.")

    def run(self, file_path: Optional[str] = None, clear_existing: bool = False):
        """
        runs the ingestion pipeline
        """
        if clear_existing:
            self._clear_data()
            if not file_path:
                return

        if not file_path:
            raise ValueError("file_path must be provided when not running a cleanup-only command.")

        print(f"Starting ingestion for {file_path}...")

        # 1. parse
        messages = self.parser.parse_file(file_path)
        print(f"Parsed {len(messages)} messages.")

        # 2. chunk
        chunks = self.chunker.chunk_messages(messages)
        print(f"Created {len(chunks)} chunks.")

        # 3. store in sql db
        chunk_models = []
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        session = self.db.get_session()
        try:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())

                model = ChunkModel(
                    id=chunk_id,
                    text=chunk.text,
                    metadata_json=json.dumps(chunk.metadata),
                )
                chunk_models.append(model)

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

        # 4. precompute embeddings + save to disk + load into vector db
        if ids:
            # Use profile embedding client if available, otherwise create default
            emb_client = self.embedding_client
            if emb_client is None:
                emb_client = EmbeddingClient()
            
            emb_path = file_path + ".embeddings.jsonl"

            embeddings = emb_client.embed_and_save_jsonl(
                ids=ids,
                texts=documents,
                out_path=emb_path,
                batch_size=128,
                show_progress=True,
            )

            self.vector_store.add_documents_with_embeddings(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                show_progress=True,
            )
            print("Saved embeddings to Vector database.")

        print("Ingestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest chat dump into database and vector store.")
    parser.add_argument("file", nargs="?", help="Path to the chat dump file (JSON or text)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before ingestion")
    parser.add_argument("--db-url", required=False, help="Database URL")
    parser.add_argument("--vec-path", required=False, help="Vector DB path")
    parser.add_argument("--collection", default="default", help="Vector collection name")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not args.file and not args.clear:
        parser.error("Please provide a chat dump file or specify --clear for cleanup.")

    if not args.db_url or not args.vec_path:
        parser.error("--db-url and --vec-path are required for ingestion")

    pipeline = IngestionPipeline(
        db_url=args.db_url,
        vector_db_path=args.vec_path,
        collection_name=args.collection,
    )
    pipeline.run(args.file, clear_existing=args.clear)