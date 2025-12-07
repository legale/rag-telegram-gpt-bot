# src/ingestion/pipeline.py

import sys
import os
import re
from typing import Optional

# add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.parser import ChatParser
from src.ingestion.chunker import MessageChunker
from src.storage.db import Database, ChunkModel
from src.storage.vector_store import VectorStore
from src.core.embedding import EmbeddingClient, create_embedding_client
from pathlib import Path
import uuid
import json
from src.core.syslog2 import *


class IngestionPipeline:
    def __init__(self, db_url: str, vector_db_path: str, collection_name: str = "default", profile_dir: Optional[str] = None):
        self.parser = ChatParser()
        self.chunker = MessageChunker()
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
            except SystemExit:
                # Already handled in create_embedding_client
                raise
            except Exception as e:
                import sys
                print(f"Error: Could not load profile config: {e}", file=sys.stderr)
                sys.exit(1)
        
        # If no profile config, use defaults from environment
        if embedding_client is None:
            generator = os.getenv("EMBEDDING_PROVIDER", "openrouter")
            model = os.getenv("EMBEDDING_MODEL")
            embedding_client = create_embedding_client(generator=generator, model=model)
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name,
            embedding_client=embedding_client,
        )
        self.db_url = db_url
        self.embedding_client = embedding_client

    def _clear_data(self):
        """clears sql and vector storage with verbose output"""
        syslog2(LOG_INFO, "clearing existing data")
        db_before = self.db.count_chunks()
        removed_db = self.db.clear()
        db_after = self.db.count_chunks()
        syslog2(LOG_INFO, "sql database cleanup", url=self.db_url, before=db_before, removed=removed_db, remaining=db_after)

        vector_before = self.vector_store.count()
        removed_vectors = self.vector_store.clear()
        vector_after = self.vector_store.count()
        syslog2(LOG_INFO, "vector store cleanup", path=self.vector_store.persist_directory, collection=self.vector_store.collection_name, before=vector_before, removed=removed_vectors, remaining=vector_after)
        syslog2(LOG_INFO, "data cleared")

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

        syslog2(LOG_INFO, "starting ingestion", file_path=file_path)

        # 1. parse
        messages = self.parser.parse_file(file_path)
        syslog2(LOG_INFO, "files parsed", messages_count=len(messages))

        # Determine chat_id from filename or default
        filename = os.path.basename(file_path)
        chat_id_match = re.search(r"telegram_dump_(-?\d+)", filename)
        chat_id = chat_id_match.group(1) if chat_id_match else "unknown_chat"
        syslog2(LOG_INFO, "identified chat_id", chat_id=chat_id)

        # 1.5. store messages in sql db
        db_messages = []
        for msg in messages:
            # Composite ID: {chat_id}_{msg_id} to ensure global uniqueness
            # Note: msg.id from parser is usually the telegram integer ID as string
            composite_id = f"{chat_id}_{msg.id}"
            db_messages.append({
                "msg_id": composite_id,
                "chat_id": chat_id,
                "ts": msg.timestamp,
                "from_id": msg.sender, # Using sender name as ID for now since parser doesn't provide user ID
                "text": msg.content
            })
        
        try:
            self.db.add_messages_batch(db_messages)
            syslog2(LOG_INFO, "messages saved to sql database", count=len(db_messages))
        except Exception as e:
            # Check if it's a unique constraint violation (optional improvement)
            syslog2(LOG_WARNING, "messages save issue (duplicates might exist)", error=str(e))
            # We continue because chunks might still need to be generated/saved
            # or maybe we should raise? For now, let's assume if messages exist we can proceed.
            # But duplicate keys will rollback the specific transaction.
            pass

        # 2. chunk
        chunks = self.chunker.chunk_messages(messages)
        syslog2(LOG_INFO, "chunks created", chunks_count=len(chunks))

        # 3. store in sql db
        chunk_models = []
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        session = self.db.get_session()
        try:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                
                # Prepare metadata (enhanced fields)
                # Convert chunk metadata to dict for metadata_json field
                meta_dict = {
                    "message_count": chunk.metadata.message_count,
                    "start_date": chunk.metadata.ts_from.isoformat(),
                    "end_date": chunk.metadata.ts_to.isoformat()
                }

                # Construct composite FKs
                msg_id_start = f"{chat_id}_{chunk.metadata.msg_id_start}"
                msg_id_end = f"{chat_id}_{chunk.metadata.msg_id_end}"

                model = ChunkModel(
                    id=chunk_id,
                    text=chunk.text,
                    metadata_json=json.dumps(meta_dict),
                    chat_id=chat_id,
                    msg_id_start=msg_id_start,
                    msg_id_end=msg_id_end,
                    ts_from=chunk.metadata.ts_from,
                    ts_to=chunk.metadata.ts_to
                )
                chunk_models.append(model)

                ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(meta_dict)

            session.add_all(chunk_models)
            session.commit()
            syslog2(LOG_INFO, "chunks saved to sql database")
        except Exception as e:
            session.rollback()
            syslog2(LOG_ERR, "sql save failed", error=str(e))
            raise
        finally:
            session.close()

        # 4. precompute embeddings + save to disk + load into vector db
        if ids:
            # Use embedding client from initialization (profile config or env defaults)
            emb_client = self.embedding_client
            if emb_client is None:
                # This should not happen if initialization was correct
                raise RuntimeError("Embedding client was not initialized. This is a bug.")
            
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
            syslog2(LOG_INFO, "embeddings saved to vector database")

        syslog2(LOG_INFO, "ingestion complete")


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