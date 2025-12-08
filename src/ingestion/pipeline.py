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

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable=None, desc=None, total=None, **kwargs):
        if iterable is None:
            return range(total) if total else []
        return iterable


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

    def clear_stage0(self) -> int:
        """Clear stage0: messages from SQL database."""
        syslog2(LOG_INFO, "clearing stage0: messages")
        deleted = self.db.clear_messages()
        syslog2(LOG_INFO, "stage0 cleared", deleted=deleted)
        return deleted

    def clear_stage1(self) -> int:
        """Clear stage1: embeddings for chunks."""
        syslog2(LOG_INFO, "clearing stage1: embeddings")
        before = self.vector_store.count()
        removed = self.vector_store.clear()
        syslog2(LOG_INFO, "stage1 cleared", before=before, removed=removed)
        return removed

    def clear_stage2(self) -> int:
        """Clear stage2: embeddings for clusters of chunk embeddings (topics_l1)."""
        syslog2(LOG_INFO, "clearing stage2: topics_l1")
        deleted = self.db.clear_topics_l1()
        syslog2(LOG_INFO, "stage2 cleared", deleted=deleted)
        return deleted

    def clear_stage3(self) -> int:
        """Clear stage3: topic names for chunk embeddings (topic_l1_id assignments)."""
        syslog2(LOG_INFO, "clearing stage3: topic_l1_id assignments")
        updated = self.db.clear_chunk_topic_l1_assignments()
        syslog2(LOG_INFO, "stage3 cleared", updated=updated)
        return updated

    def clear_stage4(self) -> int:
        """Clear stage4: topic names for cluster embeddings (topic_l2_id assignments and topics_l2)."""
        syslog2(LOG_INFO, "clearing stage4: topic_l2_id assignments and topics_l2")
        updated = self.db.clear_chunk_topic_l2_assignments()
        deleted = self.db.clear_topics_l2()
        syslog2(LOG_INFO, "stage4 cleared", updated=updated, deleted=deleted)
        return updated + deleted

    def clear_all(self):
        """Clear all stages."""
        # Clear in reverse order to maintain referential integrity
        self.clear_stage4()
        self.clear_stage3()
        self.clear_stage2()
        self.clear_stage1()
        # Clear chunks before messages (chunks reference messages)
        self.db.clear()
        self.clear_stage0()
        syslog2(LOG_INFO, "all stages cleared")

    def run_stage0(self, file_path: str):
        """Run stage0: parse and store messages/chunks."""
        if not file_path:
            raise ValueError("file_path is required for stage0")
        self.parse_and_store(file_path, clear_existing=False)

    def run_stage1(self, model: Optional[str] = None, batch_size: int = 128):
        """Run stage1: generate embeddings for chunks."""
        self.generate_embeddings(model=model, batch_size=batch_size)

    def _get_llm_client(self):
        """Get LLM client using model from profile config."""
        from src.core.llm import LLMClient
        
        model_name = None
        if self.profile_dir and self.profile_dir.exists():
            try:
                from src.bot.config import BotConfig
                config = BotConfig(self.profile_dir)
                model_name = config.current_model
            except Exception as e:
                syslog2(LOG_WARNING, "failed to load model from profile config", error=str(e))
        
        # Fallback to models.txt if no model found
        if not model_name:
            try:
                models_file = Path(__file__).parent.parent.parent / "models.txt"
                if models_file.exists():
                    with open(models_file, "r") as f:
                        model_name = f.readline().strip()
            except Exception as e:
                syslog2(LOG_WARNING, "failed to load model from models.txt", error=str(e))
        
        # Final fallback to default
        if not model_name:
            model_name = "openai/gpt-oss-20b:free"
            syslog2(LOG_INFO, "using default LLM model", model=model_name)
        
        return LLMClient(model=model_name, verbosity=0)

    def run_stage2(self, **clustering_params):
        """Run stage2: cluster chunk embeddings into topics_l1."""
        from src.ai.clustering import TopicClusterer
        
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        # Default parameters if not provided
        params = {
            'min_cluster_size': clustering_params.get('min_cluster_size', 2),
            'min_samples': clustering_params.get('min_samples', 1),
            'metric': clustering_params.get('metric', 'cosine'),
            'cluster_selection_method': clustering_params.get('cluster_selection_method', 'eom'),
            'cluster_selection_epsilon': clustering_params.get('cluster_selection_epsilon', 0.0)
        }
        
        clusterer.perform_l1_clustering(**params)

    def run_stage3(self, only_unnamed: bool = False, rebuild: bool = False):
        """Run stage3: generate topic names for L1 topics."""
        from src.ai.clustering import TopicClusterer
        
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        clusterer.name_topics(only_unnamed=only_unnamed, rebuild=rebuild, target='l1')

    def run_stage4(self, **clustering_params):
        """Run stage4: cluster L1 topics into L2 and generate names."""
        from src.ai.clustering import TopicClusterer
        
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        # Default parameters if not provided
        l2_params = {
            'min_cluster_size': clustering_params.get('min_cluster_size', 2),
            'min_samples': clustering_params.get('min_samples', 1),
            'metric': clustering_params.get('metric', 'cosine'),
            'cluster_selection_method': clustering_params.get('cluster_selection_method', 'eom'),
            'cluster_selection_epsilon': clustering_params.get('cluster_selection_epsilon', 0.0)
        }
        
        clusterer.perform_l2_clustering(**l2_params)
        clusterer.name_topics(target='l2')

    def run_all(self, file_path: str, model: Optional[str] = None, batch_size: int = 128, **clustering_params):
        """Run all stages in sequence."""
        print("Running stage0: Parse and store...")
        self.run_stage0(file_path)
        
        print("\nRunning stage1: Generate embeddings...")
        self.run_stage1(model=model, batch_size=batch_size)
        
        print("\nRunning stage2: Cluster L1 topics...")
        self.run_stage2(**clustering_params)
        
        print("\nRunning stage3: Name L1 topics...")
        self.run_stage3()
        
        print("\nRunning stage4: Cluster L2 topics and name...")
        self.run_stage4(**clustering_params)
        
        print("\nAll stages complete!")

    def parse_and_store(self, file_path: str, clear_existing: bool = False):
        """
        Parse file and store messages/chunks in SQLite database.
        
        Args:
            file_path: Path to chat dump file
            clear_existing: Whether to clear existing data before ingestion
        """
        if clear_existing:
            self._clear_data()

        if not file_path:
            raise ValueError("file_path must be provided")

        syslog2(LOG_INFO, "starting parse and store", file_path=file_path)

        # 1. parse
        print("Parsing file...")
        messages = self.parser.parse_file(file_path)
        print(f"Parsed {len(messages)} messages")
        syslog2(LOG_INFO, "files parsed", messages_count=len(messages))

        # Determine chat_id from filename or default
        filename = os.path.basename(file_path)
        chat_id_match = re.search(r"telegram_dump_(-?\d+)", filename)
        chat_id = chat_id_match.group(1) if chat_id_match else "unknown_chat"
        syslog2(LOG_INFO, "identified chat_id", chat_id=chat_id)

        # 1.5. store messages in sql db
        print("Preparing messages for database...")
        db_messages = []
        for msg in tqdm(messages, desc="  Processing messages", unit="msg"):
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
        print(f"Prepared {len(db_messages)} messages")
        
        print("Saving messages to database...")
        try:
            inserted_count = self.db.add_messages_batch(db_messages)
            skipped_count = len(db_messages) - inserted_count
            if skipped_count > 0:
                print(f"Saved {inserted_count} new messages ({skipped_count} duplicates skipped)")
            else:
                print(f"Saved {inserted_count} messages")
            syslog2(LOG_INFO, "messages saved to sql database", inserted=inserted_count, skipped=skipped_count, total=len(db_messages))
        except Exception as e:
            print(f"Error saving messages: {e}")
            syslog2(LOG_ERR, "messages save failed", error=str(e))
            raise

        # 2. chunk
        print("Creating chunks...")
        chunks = self.chunker.chunk_messages(messages)
        print(f"Created {len(chunks)} chunks")
        syslog2(LOG_INFO, "chunks created", chunks_count=len(chunks))

        # 3. store in sql db
        chunk_models = []
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        session = self.db.get_session()
        try:
            print("Preparing chunks for database...")
            for chunk in tqdm(chunks, desc="  Processing chunks", unit="chunk"):
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
            print(f"Prepared {len(chunk_models)} chunks")

            print("Saving chunks to database...")
            session.add_all(chunk_models)
            session.commit()
            print(f"Saved {len(chunk_models)} chunks")
            syslog2(LOG_INFO, "chunks saved to sql database", count=len(chunk_models))
        except Exception as e:
            session.rollback()
            print(f"Error: {e}")
            syslog2(LOG_ERR, "sql save failed", error=str(e))
            raise
        finally:
            session.close()

        print(f"\nParse and store complete: {len(chunk_models)} chunks saved")
        syslog2(LOG_INFO, "parse and store complete", chunks_count=len(chunk_models))

    def generate_embeddings(self, model: Optional[str] = None, batch_size: int = 128):
        """
        Generate embeddings for chunks without embeddings and save to vector database.
        
        Args:
            model: Embedding model to use (overrides profile config)
            batch_size: Batch size for embedding generation
        """
        syslog2(LOG_INFO, "starting embedding generation", model=model, batch_size=batch_size)
        
        # Get embedding client (use provided model or default)
        emb_client = self.embedding_client
        if model:
            # Create new client with specified model
            from src.core.embedding import create_embedding_client
            generator = os.getenv("EMBEDDING_PROVIDER", "openrouter")
            emb_client = create_embedding_client(generator=generator, model=model)
        
        if emb_client is None:
            raise RuntimeError("Embedding client was not initialized. This is a bug.")
        
        # Get all chunks from database that don't have embeddings in vector store
        session = self.db.get_session()
        try:
            all_chunks = session.query(ChunkModel).all()
            # Get existing IDs from vector store
            vector_data = self.vector_store.get_all_embeddings()
            existing_ids = set(vector_data.get("ids", []))
            
            # Filter chunks without embeddings
            chunks_to_embed = [chunk for chunk in all_chunks if chunk.id not in existing_ids]
            
            if not chunks_to_embed:
                syslog2(LOG_INFO, "all chunks already have embeddings")
                return
            
            syslog2(LOG_INFO, "chunks to embed", total=len(all_chunks), missing=len(chunks_to_embed))
            
            # Prepare data
            ids = [chunk.id for chunk in chunks_to_embed]
            documents = [chunk.text for chunk in chunks_to_embed]
            metadatas = []
            
            for chunk in chunks_to_embed:
                meta_dict = {}
                if chunk.metadata_json:
                    try:
                        meta_dict = json.loads(chunk.metadata_json)
                    except:
                        pass
                metadatas.append(meta_dict)
            
            # Generate embeddings
            embeddings = emb_client.get_embeddings_batched(
                texts=documents,
                batch_size=batch_size,
                show_progress=True
            )
            
            # Save to vector store
            self.vector_store.add_documents_with_embeddings(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                show_progress=True,
            )
            syslog2(LOG_INFO, "embeddings saved to vector database", count=len(ids))
            
        finally:
            session.close()
        
        syslog2(LOG_INFO, "embedding generation complete")

    def run(self, file_path: Optional[str] = None, clear_existing: bool = False):
        """
        Full ingestion pipeline (parse + embed) - backward compatibility.
        
        Args:
            file_path: Path to chat dump file
            clear_existing: Whether to clear existing data before ingestion
        """
        if clear_existing:
            self._clear_data()
            if not file_path:
                return

        if not file_path:
            raise ValueError("file_path must be provided when not running a cleanup-only command.")

        syslog2(LOG_INFO, "starting full ingestion pipeline", file_path=file_path)
        
        # Step 1: Parse and store
        self.parse_and_store(file_path, clear_existing=False)
        
        # Step 2: Generate embeddings
        self.generate_embeddings(batch_size=128)
        
        syslog2(LOG_INFO, "full ingestion pipeline complete")


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