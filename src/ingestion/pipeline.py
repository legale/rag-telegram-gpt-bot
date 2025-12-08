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
        
        # Load profile config - REQUIRED
        if not self.profile_dir or not self.profile_dir.exists():
            syslog2(LOG_ERR, "profile directory not found, cannot initialize ingestion pipeline")
            sys.exit(1)
        
        try:
            from src.bot.config import BotConfig
            config = BotConfig(self.profile_dir)
            
            # Check that embedding_model is explicitly set in config (not using default)
            embedding_model = config.data.get("embedding_model")
            if not embedding_model:
                syslog2(LOG_ERR, "embedding_model is not set in profile config", config_file=config.config_file)
                syslog2(LOG_ERR, "please add embedding_model parameter to config.json")
                syslog2(LOG_NOTICE, "example config.json", example='{"embedding_model": "paraphrase-multilingual-mpnet-base-v2", "embedding_generator": "local", "current_model": "openai/gpt-oss-20b:free"}')
                sys.exit(1)
            
            embedding_generator = config.embedding_generator
            
            embedding_client = create_embedding_client(
                generator=embedding_generator,
                model=embedding_model
            )
        except SystemExit:
            # Already handled in create_embedding_client
            raise
        except Exception as e:
            syslog2(LOG_ERR, "could not load profile config", error=str(e))
            sys.exit(1)
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name,
            embedding_client=embedding_client,
        )
        self.db_url = db_url
        self.embedding_client = embedding_client

    def _clear_data(self):
        """clears sql and vector storage with verbose output"""
        syslog2(LOG_NOTICE, "clearing existing data")
        db_before = self.db.count_chunks()
        removed_db = self.db.clear()
        db_after = self.db.count_chunks()
        syslog2(LOG_NOTICE, "sql database cleanup", url=self.db_url, before=db_before, removed=removed_db, remaining=db_after)

        vector_before = self.vector_store.count()
        removed_vectors = self.vector_store.clear()
        vector_after = self.vector_store.count()
        syslog2(LOG_NOTICE, "vector store cleanup", path=self.vector_store.persist_directory, collection=self.vector_store.collection_name, before=vector_before, removed=removed_vectors, remaining=vector_after)
        syslog2(LOG_NOTICE, "data cleared")

    def clear_stage0(self) -> int:
        """Clear stage0: messages from SQL database."""
        syslog2(LOG_NOTICE, "clearing stage0: messages")
        deleted = self.db.clear_messages()
        syslog2(LOG_NOTICE, "stage0 cleared", deleted=deleted)
        return deleted

    def clear_stage1(self) -> int:
        """Clear stage1: chunks from SQL database."""
        syslog2(LOG_NOTICE, "clearing stage1: chunks")
        deleted = self.db.clear()
        syslog2(LOG_NOTICE, "stage1 cleared", deleted=deleted)
        return deleted

    def clear_stage2(self) -> int:
        """Clear stage2: embeddings for chunks."""
        syslog2(LOG_NOTICE, "clearing stage2: embeddings")
        before = self.vector_store.count()
        removed = self.vector_store.clear()
        syslog2(LOG_NOTICE, "stage2 cleared", before=before, removed=removed)
        return removed

    def clear_stage3(self) -> int:
        """Clear stage3: L1 clustering results (topics_l1)."""
        syslog2(LOG_NOTICE, "clearing stage3: topics_l1")
        deleted = self.db.clear_topics_l1()
        syslog2(LOG_NOTICE, "stage3 cleared", deleted=deleted)
        return deleted

    def clear_stage4(self) -> int:
        """Clear stage4: topic_l1_id assignments in chunks."""
        syslog2(LOG_NOTICE, "clearing stage4: topic_l1_id assignments")
        updated = self.db.clear_chunk_topic_l1_assignments()
        syslog2(LOG_NOTICE, "stage4 cleared", updated=updated)
        return updated

    def clear_stage5(self) -> int:
        """Clear stage5: topic_l2_id assignments and topics_l2."""
        syslog2(LOG_NOTICE, "clearing stage5: topic_l2_id assignments and topics_l2")
        updated = self.db.clear_chunk_topic_l2_assignments()
        deleted = self.db.clear_topics_l2()
        syslog2(LOG_NOTICE, "stage5 cleared", updated=updated, deleted=deleted)
        return updated + deleted

    def clear_all(self):
        """Clear all stages."""
        # Clear in reverse order to maintain referential integrity
        self.clear_stage5()
        self.clear_stage4()
        self.clear_stage3()
        self.clear_stage2()
        self.clear_stage1()
        self.clear_stage0()
        syslog2(LOG_NOTICE, "all stages cleared")

    def run_stage0(self, file_path: str):
        """Run stage0: parse and store messages."""
        if not file_path:
            raise ValueError("file_path is required for stage0")
        self.parse_and_store_messages(file_path)

    def run_stage1(self, chunk_size: Optional[int] = None):
        """Run stage1: create and store chunks."""
        # If chunk_size not provided, get from profile config
        if chunk_size is None:
            if not self.profile_dir or not self.profile_dir.exists():
                import sys
                syslog2(LOG_ERR, "profile directory not found")
                sys.exit(1)
            
            from src.bot.config import BotConfig
            config = BotConfig(self.profile_dir)
            chunk_size = config.chunk_size
        
        self.parse_and_store_chunks(chunk_size=chunk_size)

    def run_stage2(self, model: Optional[str] = None, batch_size: int = 128):
        """Run stage2: generate embeddings for chunks."""
        self.generate_embeddings(model=model, batch_size=batch_size)

    def _get_llm_client(self):
        """Get LLM client using model from profile config. Model must be explicitly set."""
        from src.core.llm import LLMClient
        import sys
        
        if not self.profile_dir or not self.profile_dir.exists():
            syslog2(LOG_ERR, "profile directory not found")
            sys.exit(1)
        
        try:
            from src.bot.config import BotConfig
            config = BotConfig(self.profile_dir)
            
            # Check that current_model is explicitly set in config
            model_name = config.data.get("current_model")
            if not model_name:
                syslog2(LOG_ERR, "current_model is not set in profile config", config_file=config.config_file)
                syslog2(LOG_ERR, "please add current_model parameter to config.json")
                syslog2(LOG_NOTICE, "example config.json", example='{"embedding_model": "paraphrase-multilingual-mpnet-base-v2", "embedding_generator": "local", "current_model": "openai/gpt-oss-20b:free"}')
                sys.exit(1)
            
            return LLMClient(model=model_name, verbosity=0)
        except Exception as e:
            syslog2(LOG_ERR, "failed to load model from profile config", error=str(e))
            sys.exit(1)

    def run_stage3(self, **clustering_params):
        """Run stage3: cluster chunk embeddings into topics_l1 (HDBSCAN clustering)."""
        from src.ai.clustering import TopicClusterer
        
        # LLM client not needed for clustering - only for naming (stage4)
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=None
        )
        
        # Default parameters if not provided
        params = {
            'min_cluster_size': clustering_params.get('min_cluster_size', 2),
            'min_samples': clustering_params.get('min_samples', 1),
            'metric': clustering_params.get('metric', 'cosine'),
            'cluster_selection_method': clustering_params.get('cluster_selection_method', 'eom'),
            'cluster_selection_epsilon': clustering_params.get('cluster_selection_epsilon', 0.0)
        }
        
        # Perform clustering and get assignments
        assignments = clusterer.perform_l1_clustering(**params)
        # Store assignments for stage4
        self._stage3_assignments = assignments

    def run_stage4(self, only_unnamed: bool = True, rebuild: bool = False):
        """Run stage4: create embeddings for clusters (centroids), assign topic_l1_id to chunks, and name L1 topics."""
        from src.ai.clustering import TopicClusterer
        import numpy as np
        import json
        
        # LLM client needed for naming
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        # Check if assignments are in memory (from stage3)
        if hasattr(self, '_stage3_assignments'):
            # Use assignments from stage3
            clusterer._l1_topic_assignments = self._stage3_assignments
            clusterer.assign_l1_topics_to_chunks()
            # Clear assignments after use
            delattr(self, '_stage3_assignments')
        else:
            # Restore assignments from database: topics_l1 exist but chunks don't have topic_l1_id yet
            l1_topics = self.db.get_all_topics_l1()
            if not l1_topics:
                raise ValueError("stage4 requires stage3 to be run first (no topics_l1 found)")
            
            # Check if chunks already have topic_l1_id assigned
            session = self.db.get_session()
            chunks_already_assigned = False
            try:
                from src.storage.db import ChunkModel
                chunks_with_topics = session.query(ChunkModel).filter(ChunkModel.topic_l1_id.isnot(None)).count()
                if chunks_with_topics > 0:
                    chunks_already_assigned = True
                    syslog2(LOG_NOTICE, "chunks already have topic_l1_id assigned, skipping assignment step")
            finally:
                session.close()
            
            # Only restore assignments if chunks are not already assigned
            if not chunks_already_assigned:
                # Restore assignments by finding nearest topic for each chunk based on centroids
                syslog2(LOG_NOTICE, "restoring l1 topic assignments from database centroids")
                
                # Get all topics with centroids
                topic_centroids = {}
                for topic in l1_topics:
                    if topic.center_vec:
                        try:
                            centroid = json.loads(topic.center_vec)
                            topic_centroids[topic.id] = np.array(centroid)
                        except:
                            pass
                
                if not topic_centroids:
                    raise ValueError("no topic centroids found in topics_l1")
                
                # Get all chunks with embeddings
                all_chunks = self.vector_store.get_all_embeddings()
                if not all_chunks or not all_chunks.get('ids'):
                    raise ValueError("no chunks with embeddings found")
                
                # Build assignments: topic_id -> [chunk_ids]
                assignments = {}
                for topic_id in topic_centroids.keys():
                    assignments[topic_id] = []
                
                # For each chunk, find nearest topic by cosine similarity
                chunk_ids = all_chunks['ids']
                embeddings = all_chunks['embeddings']
                metadatas = all_chunks.get('metadatas', [None] * len(chunk_ids))
                
                total_chunks = len(chunk_ids)
                syslog2(LOG_NOTICE, "finding nearest topics for chunks", total_chunks=total_chunks)
                
                try:
                    pbar = tqdm(total=total_chunks, desc="Matching chunks to topics", unit="chunk")
                except ImportError:
                    pbar = None
                
                try:
                    for chunk_id, embedding, metadata in zip(chunk_ids, embeddings, metadatas):
                        chunk_emb = np.array(embedding)
                        
                        # Calculate cosine similarity to each topic centroid
                        best_topic_id = None
                        best_similarity = -1.0
                        
                        for topic_id, centroid in topic_centroids.items():
                            # Cosine similarity
                            similarity = np.dot(chunk_emb, centroid) / (np.linalg.norm(chunk_emb) * np.linalg.norm(centroid))
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_topic_id = topic_id
                        
                        if best_topic_id:
                            assignments[best_topic_id].append(chunk_id)
                        
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({"topics": len([a for a in assignments.values() if a])})
                finally:
                    if pbar:
                        pbar.close()
                
                # Assign topics to chunks
                clusterer._l1_topic_assignments = assignments
                clusterer.assign_l1_topics_to_chunks(show_progress=True)
        
        # Name L1 topics (after assignments are done)
        # Create progress callback for topic naming
        def progress_callback(current, total, stage, total_all=None):
            try:
                from tqdm import tqdm
                if not hasattr(progress_callback, 'pbar'):
                    desc = f"Naming {stage.upper()} topics"
                    progress_callback.pbar = tqdm(total=total, desc=desc, unit="topic")
                progress_callback.pbar.update(1)
                if total_all is not None and total_all != total:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total} filtered/total: {total}/{total_all}"})
                elif total_all is not None:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total} (all: {total_all})"})
                else:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total}"})
                if current == total:
                    progress_callback.pbar.close()
                    delattr(progress_callback, 'pbar')
            except ImportError:
                if total_all is not None and total_all != total:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, filtered=total, total=total_all)
                elif total_all is not None:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, total=total, all=total_all)
                else:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, total=total)
        
        syslog2(LOG_NOTICE, "naming l1 topics")
        clusterer.name_topics(
            progress_callback=progress_callback,
            only_unnamed=only_unnamed,
            rebuild=rebuild,
            target='l1'
        )

    def run_stage5(self, **clustering_params):
        """Run stage5: cluster L1 topics into L2 and generate names."""
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
        
        syslog2(LOG_NOTICE, "clustering l1 topics into l2 topics")
        clusterer.perform_l2_clustering(**l2_params)
        
        # Create progress callback for topic naming
        def progress_callback(current, total, stage, total_all=None):
            try:
                from tqdm import tqdm
                if not hasattr(progress_callback, 'pbar'):
                    desc = f"Naming {stage.upper()} topics"
                    progress_callback.pbar = tqdm(total=total, desc=desc, unit="topic")
                progress_callback.pbar.update(1)
                if total_all is not None and total_all != total:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total} filtered/total: {total}/{total_all}"})
                elif total_all is not None:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total} (all: {total_all})"})
                else:
                    progress_callback.pbar.set_postfix({"progress": f"{current}/{total}"})
                if current == total:
                    progress_callback.pbar.close()
                    delattr(progress_callback, 'pbar')
            except ImportError:
                if total_all is not None and total_all != total:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, filtered=total, total=total_all)
                elif total_all is not None:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, total=total, all=total_all)
                else:
                    syslog2(LOG_NOTICE, f"naming {stage} topics", current=current, total=total)
        
        clusterer.name_topics(progress_callback=progress_callback, only_unnamed=True, target='l2')

    def run_all(self, file_path: str, chunk_size: Optional[int] = None, model: Optional[str] = None, batch_size: int = 128, **clustering_params):
        """Run all stages in sequence."""
        syslog2(LOG_NOTICE, "running stage0: parse and store messages")
        self.run_stage0(file_path)
        
        syslog2(LOG_NOTICE, "running stage1: create and store chunks")
        # chunk_size will be taken from config in run_stage1 if None
        self.run_stage1(chunk_size=chunk_size)
        
        syslog2(LOG_NOTICE, "running stage2: generate embeddings")
        self.run_stage2(model=model, batch_size=batch_size)
        
        syslog2(LOG_NOTICE, "running stage3: cluster l1 topics")
        self.run_stage3(**clustering_params)
        
        syslog2(LOG_NOTICE, "running stage4: create embeddings for clusters, assign topics, and name l1 topics")
        self.run_stage4()
        
        syslog2(LOG_NOTICE, "running stage5: cluster l2 topics and name")
        self.run_stage5(**clustering_params)
        
        syslog2(LOG_NOTICE, "all stages complete")

    def parse_and_store_messages(self, file_path: str):
        """
        Parse file and store messages in SQLite database.
        
        Args:
            file_path: Path to chat dump file
        """
        if not file_path:
            raise ValueError("file_path must be provided")

        syslog2(LOG_NOTICE, "starting parse and store messages", file_path=file_path)

        # Parse file
        syslog2(LOG_NOTICE, "parsing file", file_path=file_path)
        messages = self.parser.parse_file(file_path)
        syslog2(LOG_NOTICE, "file parsed", messages_count=len(messages))
        syslog2(LOG_NOTICE, "files parsed", messages_count=len(messages))

        # Determine chat_id from filename or default
        filename = os.path.basename(file_path)
        chat_id_match = re.search(r"telegram_dump_(-?\d+)", filename)
        chat_id = chat_id_match.group(1) if chat_id_match else "unknown_chat"
        syslog2(LOG_NOTICE, "identified chat_id", chat_id=chat_id)

        # Store messages in sql db
        syslog2(LOG_NOTICE, "preparing messages for database")
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
        syslog2(LOG_NOTICE, "messages prepared for database", count=len(db_messages))
        
        syslog2(LOG_NOTICE, "saving messages to database")
        try:
            inserted_count = self.db.add_messages_batch(db_messages)
            skipped_count = len(db_messages) - inserted_count
            if skipped_count > 0:
                syslog2(LOG_NOTICE, "messages saved", inserted=inserted_count, skipped=skipped_count)
            else:
                syslog2(LOG_NOTICE, "messages saved", inserted=inserted_count)
            syslog2(LOG_NOTICE, "messages saved to sql database", inserted=inserted_count, skipped=skipped_count, total=len(db_messages))
        except Exception as e:
            syslog2(LOG_ERR, "error saving messages", error=str(e))
            syslog2(LOG_ERR, "messages save failed", error=str(e))
            raise

        syslog2(LOG_NOTICE, "stage0 complete", messages_saved=inserted_count)
        syslog2(LOG_NOTICE, "stage0 complete", messages_count=inserted_count)

    def parse_and_store_chunks(self, chunk_size: int = 6):
        """
        Create chunks from messages and store in SQLite database.
        
        Args:
            chunk_size: Number of messages per chunk (default: 10)
        """
        syslog2(LOG_NOTICE, "starting parse and store chunks", chunk_size=chunk_size)

        # Get all messages from database
        session = self.db.get_session()
        try:
            from src.storage.db import MessageModel
            messages_db = session.query(MessageModel).order_by(MessageModel.ts).all()
            
            if not messages_db:
                import sys
                syslog2(LOG_ERR, "no messages found in database, run ingest stage0 first")
                sys.exit(1)
            
            # Convert to ChatMessage format for chunker
            from src.ingestion.parser import ChatMessage
            messages = []
            for msg_db in messages_db:
                # Extract original msg_id from composite_id (remove chat_id prefix)
                original_id = msg_db.msg_id.split('_', 1)[1] if '_' in msg_db.msg_id else msg_db.msg_id
                messages.append(ChatMessage(
                    id=original_id,
                    timestamp=msg_db.ts,
                    sender=msg_db.from_id or "Unknown",
                    content=msg_db.text
                ))
            
            # Get chat_id from first message
            chat_id = messages_db[0].chat_id if messages_db else "unknown_chat"
        finally:
            session.close()

        syslog2(LOG_NOTICE, "found messages in database", count=len(messages))

        # Update chunker with new chunk_size
        self.chunker = MessageChunker(max_messages_per_chunk=chunk_size, overlap=0)

        # Create chunks
        syslog2(LOG_NOTICE, "creating chunks", chunk_size=chunk_size)
        chunks = self.chunker.chunk_messages(messages)
        syslog2(LOG_NOTICE, "chunks created", count=len(chunks))
        syslog2(LOG_NOTICE, "chunks created", chunks_count=len(chunks), chunk_size=chunk_size)

        # Store chunks in sql db
        chunk_models = []
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        session = self.db.get_session()
        try:
            syslog2(LOG_NOTICE, "preparing chunks for database")
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
            syslog2(LOG_NOTICE, "chunks prepared for database", count=len(chunk_models))

            syslog2(LOG_NOTICE, "saving chunks to database")
            session.add_all(chunk_models)
            session.commit()
            syslog2(LOG_NOTICE, "chunks saved", count=len(chunk_models))
            syslog2(LOG_NOTICE, "chunks saved to sql database", count=len(chunk_models))
        except Exception as e:
            session.rollback()
            syslog2(LOG_ERR, "error saving chunks", error=str(e))
            syslog2(LOG_ERR, "sql save failed", error=str(e))
            raise
        finally:
            session.close()

        syslog2(LOG_NOTICE, "stage1 complete", chunks_saved=len(chunk_models))
        syslog2(LOG_NOTICE, "stage1 complete", chunks_count=len(chunk_models))

    def parse_and_store(self, file_path: str, clear_existing: bool = False):
        """
        Parse file and store messages/chunks in SQLite database.
        Legacy method for backward compatibility - calls stage0 and stage1.
        
        Args:
            file_path: Path to chat dump file
            clear_existing: Whether to clear existing data before ingestion
        """
        if clear_existing:
            self._clear_data()
        
        self.parse_and_store_messages(file_path)
        self.parse_and_store_chunks()

    def generate_embeddings(self, model: Optional[str] = None, batch_size: int = 128):
        """
        Generate embeddings for chunks without embeddings and save to vector database.
        
        Args:
            model: Embedding model to use (overrides profile config)
            batch_size: Batch size for embedding generation
        """
        syslog2(LOG_NOTICE, "starting embedding generation", model=model, batch_size=batch_size)
        
        # Get embedding client (use provided model or default)
        emb_client = self.embedding_client
        if model:
            # Create new client with specified model
            from src.core.embedding import create_embedding_client
            generator = os.getenv("EMBEDDING_PROVIDER", "openrouter")
            emb_client = create_embedding_client(generator=generator, model=model)
        
        if emb_client is None:
            raise RuntimeError("Embedding client was not initialized. This is a bug.")
        
        # Get expected dimension from embedding client
        new_dimension = emb_client.get_dimension()
        
        # CRITICAL: Check and fix collection dimension BEFORE generating embeddings
        # This prevents ChromaDB errors when trying to add embeddings with wrong dimension
        try:
            collection_count = self.vector_store.collection.count()
            if collection_count > 0:
                # Collection has data - check its dimension
                sample = self.vector_store.collection.get(limit=1, include=["embeddings"])
                if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                    existing_dim = len(sample["embeddings"][0])
                    if existing_dim != new_dimension:
                        # Dimension mismatch - MUST recreate collection before proceeding
                        syslog2(LOG_WARNING, 
                            f"Collection dimension mismatch detected: existing={existing_dim}, new={new_dimension}. "
                            f"Recreating collection before generating embeddings...")
                        self.vector_store.collection = self.vector_store._recreate_collection_with_dimension(new_dimension)
                        self.vector_store.expected_dimension = new_dimension
                        syslog2(LOG_NOTICE, f"Collection recreated with dimension {new_dimension}")
        except Exception as e:
            syslog2(LOG_DEBUG, f"Could not check collection dimension: {e}")
        
        # Update expected dimension
        self.vector_store.expected_dimension = new_dimension
        
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
                syslog2(LOG_NOTICE, "all chunks already have embeddings")
                return
            
            syslog2(LOG_NOTICE, "chunks to embed", total=len(all_chunks), missing=len(chunks_to_embed))
            
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
                # Add topic assignments to metadata for chroma_db filtering
                if chunk.topic_l1_id is not None:
                    meta_dict["topic_l1_id"] = chunk.topic_l1_id
                if chunk.topic_l2_id is not None:
                    meta_dict["topic_l2_id"] = chunk.topic_l2_id
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
            syslog2(LOG_NOTICE, "embeddings saved to vector database", count=len(ids))
            
        finally:
            session.close()
        
        syslog2(LOG_NOTICE, "embedding generation complete")

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

        syslog2(LOG_NOTICE, "starting full ingestion pipeline", file_path=file_path)
        
        # Step 1: Parse and store
        self.parse_and_store(file_path, clear_existing=False)
        
        # Step 2: Generate embeddings
        self.generate_embeddings(batch_size=128)
        
        syslog2(LOG_NOTICE, "full ingestion pipeline complete")

    # ========================================================================
    # Topic Management Methods
    # ========================================================================

    def list_topics(self) -> None:
        """
        List all topics (L1 and L2) in a hierarchical structure with formatted table output.
        """
        try:
            l2_topics = self.db.get_all_topics_l2()
            l1_topics = self.db.get_all_topics_l1()
            
            if not l1_topics and not l2_topics:
                print("No topics found. Run 'legale ingest stage3' or 'legale topics build' first.")
                return
            
            # Group L1 topics by L2 parent
            l1_by_l2 = {}
            orphans = []
            for t in l1_topics:
                if t.parent_l2_id:
                    if t.parent_l2_id not in l1_by_l2:
                        l1_by_l2[t.parent_l2_id] = []
                    l1_by_l2[t.parent_l2_id].append(t)
                else:
                    orphans.append(t)
            
            # Print header
            print()
            print(f"{'ID':<5} {'L2 Topic Title':<40} {'L1 Count':<10} {'Chunks':<10}")
            print("-" * 75)
            
            # Show L2 topics with their L1 children
            if l2_topics:
                for l2 in l2_topics:
                    children = l1_by_l2.get(l2.id, [])
                    chunks_count = sum(c.chunk_count for c in children)
                    title = (l2.title or "unknown")[:38]  # Truncate if too long
                    
                    # Print L2 topic header
                    print(f"{l2.id:<5} {title:<40} {len(children):<10} {chunks_count:<10}")
                    
                    # Show L1 topics under each L2 topic
                    if children:
                        for l1 in children:
                            l1_title = (l1.title or "unknown")[:38]  # Truncate if too long
                            print(f"  └─ {l1.id:<3} {l1_title:<36} {l1.chunk_count:>8} chunks")
            
            # Show orphaned L1 topics
            if orphans:
                print()
                print("Orphaned L1 Topics (No Super-Topic):")
                print("-" * 75)
                for t in orphans:
                    title = (t.title or "unknown")[:38]  # Truncate if too long
                    print(f"{t.id:<5} {title:<40} {t.chunk_count:>8}")
            
            # If no L2 topics but L1 topics exist, show all L1 topics
            if not l2_topics and l1_topics:
                for t in l1_topics:
                    title = (t.title or "unknown")[:38]  # Truncate if too long
                    print(f"{t.id:<5} {title:<40} {t.chunk_count:>8}")
                
        except Exception as e:
            syslog2(LOG_ERR, "error listing topics", error=str(e))
            print(f"Error listing topics: {e}")

    def show_topic(self, topic_id: int) -> None:
        """
        Show detailed information about a topic (L1 or L2).
        
        Args:
            topic_id: Topic ID to show
        """
        try:
            # Try L2 first
            l2 = next((t for t in self.db.get_all_topics_l2() if t.id == topic_id), None)
            
            if l2:
                print(f"=== Super-Topic L2-{l2.id} ===")
                print(f"Title: {l2.title}")
                print(f"Description: {l2.descr}")
                print(f"Chunks: {l2.chunk_count}")
                
                subtopics = self.db.get_l1_topics_by_l2(l2.id)
                print(f"\nSub-topics ({len(subtopics)}):")
                for sub in subtopics:
                    print(f"  L1-{sub.id}: {sub.title} ({sub.chunk_count} chunks)")
                return

            # Try L1
            l1 = next((t for t in self.db.get_all_topics_l1() if t.id == topic_id), None)
            if l1:
                print(f"=== Topic L1-{l1.id} ===")
                print(f"Title: {l1.title}")
                print(f"Description: {l1.descr}")
                print(f"Parent L2: {l1.parent_l2_id}")
                print(f"Chunks: {l1.chunk_count}")
                print(f"Messages: {l1.msg_count}")
                print(f"Time: {l1.ts_from} - {l1.ts_to}")
                
                chunks = self.db.get_chunks_by_topic_l1(l1.id)
                print(f"\nSample Content ({min(3, len(chunks))} of {len(chunks)}):")
                for i, c in enumerate(chunks[:3]):
                    print(f"--- Chunk {i+1} ---")
                    print(c.text[:200].replace('\n', ' ') + "...")
                return
                
            print(f"Topic ID {topic_id} not found in L1 or L2 tables.")
                
        except ValueError:
            print(f"Error: topic id must be an integer")
        except Exception as e:
            print(f"Error showing topic: {e}")
            syslog2(LOG_ERR, "error showing topic", error=str(e))


if __name__ == "__main__":
    from src.core.cli_parser import (
        CommandParser, CommandSpec, ArgStream, CLIError, CLIHelp,
        parse_option, parse_flag
    )
    
    def parse_ingest_main(stream: ArgStream) -> dict:
        """Parse ingest command for pipeline main."""
        file = None
        if stream.has_next() and not stream.peek().startswith("--"):
            file = stream.next()
        clear = parse_flag(stream, "clear")
        db_url = parse_option(stream, "db-url")
        vec_path = parse_option(stream, "vec-path")
        collection = parse_option(stream, "collection") or "default"
        return {
            "file": file,
            "clear": clear,
            "db_url": db_url,
            "vec_path": vec_path,
            "collection": collection
        }
    
    commands = [
        CommandSpec("ingest", parse_ingest_main, "Ingest chat dump into database and vector store"),
    ]
    
    parser = CommandParser(commands)
    
    if len(sys.argv) == 1:
        print("Usage: python -m src.ingestion.pipeline ingest [file] [--clear] [--db-url <url>] [--vec-path <path>] [--collection <name>]", file=sys.stderr)
        sys.exit(1)
    
    try:
        cmd_name, args = parser.parse(sys.argv[1:])
    except CLIHelp:
        print("Ingest chat dump into database and vector store")
        print("\nUsage: python -m src.ingestion.pipeline ingest [file] [--clear] [--db-url <url>] [--vec-path <path>] [--collection <name>]")
        sys.exit(0)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not args.file and not args.clear:
        print("Error: Please provide a chat dump file or specify --clear for cleanup.", file=sys.stderr)
        sys.exit(1)
    
    if not args.db_url or not args.vec_path:
        print("Error: --db-url and --vec-path are required for ingestion", file=sys.stderr)
        sys.exit(1)
    
    pipeline = IngestionPipeline(
        db_url=args.db_url,
        vector_db_path=args.vec_path,
        collection_name=args.collection,
    )
    pipeline.run(args.file, clear_existing=args.clear)
    pipeline.run(args.file, clear_existing=args.clear)