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
    def __init__(self, db_url: str, vector_db_path: str, collection_name: str = "embed-l1", profile_dir: Optional[str] = None):
        self.parser = ChatParser()
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
            
            # Initialize chunker with token-based parameters from config
            self.chunker = MessageChunker(
                chunk_token_min=config.chunk_token_min,
                chunk_token_max=config.chunk_token_max,
                chunk_overlap_ratio=config.chunk_overlap_ratio
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
        """Clear stage2: embeddings for chunks (embedding_json in SQLite)."""
        syslog2(LOG_NOTICE, "clearing stage2: chunk embeddings")
        session = self.db.get_session()
        try:
            updated = session.query(ChunkModel).update({
                ChunkModel.embedding_json: None,
                ChunkModel.embedding_dim: None
            }, synchronize_session=False)
            session.commit()
            syslog2(LOG_NOTICE, "stage2 cleared", updated=updated)
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_stage3(self) -> int:
        """Clear stage3: chunks from vector_db collection."""
        syslog2(LOG_NOTICE, "clearing stage3: vector_db chunks")
        before = self.vector_store.count()
        removed = self.vector_store.clear()
        syslog2(LOG_NOTICE, "stage3 cleared", before=before, removed=removed)
        return removed

    def clear_stage4(self) -> int:
        """Clear stage4: L1 clustering results (topics_l1) and topic_l1_id assignments."""
        syslog2(LOG_NOTICE, "clearing stage4: topics_l1 and assignments")
        updated = self.db.clear_chunk_topic_l1_assignments()
        deleted = self.db.clear_topics_l1()
        syslog2(LOG_NOTICE, "stage4 cleared", updated=updated, deleted=deleted)
        return updated + deleted

    def clear_stage5(self) -> int:
        """Clear stage5: L1 topics from vector_db collection."""
        syslog2(LOG_NOTICE, "clearing stage5: vector_db topics_l1")
        before = self.vector_store.topics_l1_collection.count()
        if before > 0:
            all_data = self.vector_store.topics_l1_collection.get()
            if all_data and all_data.get("ids"):
                self.vector_store.topics_l1_collection.delete(ids=all_data["ids"])
        after = self.vector_store.topics_l1_collection.count()
        removed = before - after
        syslog2(LOG_NOTICE, "stage5 cleared", before=before, removed=removed)
        return removed

    def clear_stage6(self) -> int:
        """Clear stage6: L2 clustering results (topics_l2) and topic_l2_id assignments."""
        syslog2(LOG_NOTICE, "clearing stage6: topics_l2 and assignments")
        updated = self.db.clear_chunk_topic_l2_assignments()
        deleted = self.db.clear_topics_l2()
        syslog2(LOG_NOTICE, "stage6 cleared", updated=updated, deleted=deleted)
        return updated + deleted

    def clear_stage7(self) -> int:
        """Clear stage7: L2 topics from vector_db collection."""
        syslog2(LOG_NOTICE, "clearing stage7: vector_db topics_l2")
        before = self.vector_store.topics_l2_collection.count()
        if before > 0:
            all_data = self.vector_store.topics_l2_collection.get()
            if all_data and all_data.get("ids"):
                self.vector_store.topics_l2_collection.delete(ids=all_data["ids"])
        after = self.vector_store.topics_l2_collection.count()
        removed = before - after
        syslog2(LOG_NOTICE, "stage7 cleared", before=before, removed=removed)
        return removed

    def clear_stage8(self) -> int:
        """Clear stage8: L1 topic names (reset to 'unknown')."""
        syslog2(LOG_NOTICE, "clearing stage8: l1 topic names")
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL1Model
            updated = session.query(TopicL1Model).update({
                TopicL1Model.title: "unknown",
                TopicL1Model.descr: "Pending description..."
            }, synchronize_session=False)
            session.commit()
            syslog2(LOG_NOTICE, "stage8 cleared", updated=updated)
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_stage9(self) -> int:
        """Clear stage9: L2 topic names (reset to 'unknown')."""
        syslog2(LOG_NOTICE, "clearing stage9: l2 topic names")
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL2Model
            updated = session.query(TopicL2Model).update({
                TopicL2Model.title: "unknown",
                TopicL2Model.descr: "Pending description..."
            }, synchronize_session=False)
            session.commit()
            syslog2(LOG_NOTICE, "stage9 cleared", updated=updated)
            return updated
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def clear_all(self):
        """Clear all stages."""
        # Clear in reverse order to maintain referential integrity
        self.clear_stage9()
        self.clear_stage8()
        self.clear_stage7()
        self.clear_stage6()
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

    def run_stage1(self):
        """Run stage1: create and store chunks."""
        if not self.profile_dir or not self.profile_dir.exists():
            import sys
            syslog2(LOG_ERR, "profile directory not found")
            sys.exit(1)
        
        self.parse_and_store_chunks()

    def run_stage2(self, model: Optional[str] = None, batch_size: int = 128):
        """Run stage2: generate embeddings for chunks and save to SQLite (chunks.embedding_json)."""
        self.generate_embeddings(model=model, batch_size=batch_size)

    def run_stage3(self):
        """Run stage3: sync chunks from SQLite (embedding_json) to vector_db."""
        syslog2(LOG_NOTICE, "starting stage3: syncing chunks to vector database")
        
        # Get all chunks with embeddings from SQLite
        session = self.db.get_session()
        try:
            chunks_with_embeddings = session.query(ChunkModel).filter(
                ChunkModel.embedding_json.isnot(None)
            ).all()
            
            if not chunks_with_embeddings:
                syslog2(LOG_NOTICE, "no chunks with embeddings found in sqlite")
                return
            
            total_chunks = len(chunks_with_embeddings)
            syslog2(LOG_NOTICE, "syncing chunks to vector database", total=total_chunks)
            
            # Prepare data for vector store
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks_with_embeddings:
                # Parse embedding from JSON
                try:
                    embedding = json.loads(chunk.embedding_json)
                except (json.JSONDecodeError, TypeError) as e:
                    syslog2(LOG_WARNING, "failed to parse embedding_json for chunk", chunk_id=chunk.id, error=str(e))
                    continue
                
                ids.append(chunk.id)
                documents.append(chunk.text)
                embeddings.append(embedding)
                
                # Prepare metadata
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
            
            if not ids:
                syslog2(LOG_WARNING, "no valid embeddings found to sync")
                return
            
            # Check collection dimension before syncing
            new_dimension = len(embeddings[0]) if embeddings else 0
            try:
                collection_count = self.vector_store.collection.count()
                if collection_count > 0:
                    # Collection has data - check its dimension
                    sample = self.vector_store.collection.get(limit=1, include=["embeddings"])
                    if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                        existing_dim = len(sample["embeddings"][0])
                        if existing_dim != new_dimension:
                            # Dimension mismatch - MUST recreate collection
                            syslog2(LOG_WARNING, 
                                f"Collection dimension mismatch: existing={existing_dim}, new={new_dimension}. "
                                f"Recreating collection...")
                            self.vector_store.collection = self.vector_store._recreate_collection_with_dimension(new_dimension)
                            self.vector_store.expected_dimension = new_dimension
                            syslog2(LOG_NOTICE, f"Collection recreated with dimension {new_dimension}")
            except Exception as e:
                syslog2(LOG_DEBUG, f"Could not check collection dimension: {e}")
            
            # Update expected dimension
            self.vector_store.expected_dimension = new_dimension
            
            # Sync to vector store (add or update)
            # ChromaDB will update existing IDs automatically
            self.vector_store.add_documents_with_embeddings(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                show_progress=True,
            )
            syslog2(LOG_NOTICE, "chunks synced to vector database", count=len(ids))
            
        finally:
            session.close()
        
        syslog2(LOG_NOTICE, "stage3 complete")

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
            
            return LLMClient(model=model_name, log_level=LOG_WARNING)
        except Exception as e:
            syslog2(LOG_ERR, "failed to load model from profile config", error=str(e))
            sys.exit(1)

    def run_stage4(self, **clustering_params):
        """Run stage4: L1 clustering - cluster chunk embeddings into topics_l1 (HDBSCAN clustering).
        Reads embeddings from SQLite, saves center_vec_json to SQLite, does NOT write to vector_db."""
        from src.ai.clustering import TopicClusterer
        
        # LLM client not needed for clustering - only for naming (stage8)
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
        # Assign topics to chunks
        clusterer.assign_l1_topics_to_chunks(show_progress=True)
        # Store assignments for potential future use (though they're already assigned)
        self._stage4_assignments = assignments

    def run_stage5(self):
        """Run stage5: sync L1 topics from SQLite (center_vec_json) to vector_db."""
        syslog2(LOG_NOTICE, "starting stage5: syncing l1 topics to vector database")
        
        # Get all L1 topics with center_vec_json from SQLite
        l1_topics = self.db.get_all_topics_l1()
        
        if not l1_topics:
            syslog2(LOG_NOTICE, "no l1 topics found in sqlite")
            return
        
        # Prepare data for vector store
        ids = []
        embeddings = []
        metadatas = []
        
        for topic in l1_topics:
            if not topic.center_vec_json:
                syslog2(LOG_WARNING, "l1 topic missing center_vec_json", topic_id=topic.id)
                continue
            
            # Parse center_vec from JSON
            try:
                center_vec = json.loads(topic.center_vec_json)
            except (json.JSONDecodeError, TypeError) as e:
                syslog2(LOG_WARNING, "failed to parse center_vec_json for l1 topic", topic_id=topic.id, error=str(e))
                continue
            
            l1_topic_id = f"l1-{topic.id}"
            ids.append(l1_topic_id)
            embeddings.append(center_vec)
            metadatas.append({
                "topic_l1_id": topic.id,
                "title": topic.title or "unknown",
                "chunk_count": topic.chunk_count,
                "msg_count": topic.msg_count
            })
        
        if not ids:
            syslog2(LOG_WARNING, "no valid l1 topic centroids found to sync")
            return
        
        # Sync to vector store (add or update)
        # ChromaDB will update existing IDs automatically
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.vector_store.topics_l1_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            if i % (batch_size * 10) == 0:
                print(f"\rSyncing L1 topics: {min(i + batch_size, len(ids))}/{len(ids)}", flush=True, end="")
        
        print()  # Newline after progress
        syslog2(LOG_NOTICE, "l1 topics synced to vector database", count=len(ids))
        syslog2(LOG_NOTICE, "stage5 complete")

    def run_stage8(self, only_unnamed: bool = True, rebuild: bool = False):
        """Run stage8: name L1 topics using LLM."""
        from src.ai.clustering import TopicClusterer
        
        # LLM client needed for naming
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        # Create progress callback for topic naming
        def progress_callback(current, total, stage, total_all=None):
            percentage = int((current / total * 100)) if total > 0 else 0
            if total_all is not None and total_all != total:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%) (filtered/total: {total}/{total_all})", flush=True, end="")
            elif total_all is not None:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%) (all: {total_all})", flush=True, end="")
            else:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%)", flush=True, end="")
            if current == total:
                print()  # Newline after progress
        
        syslog2(LOG_NOTICE, "naming l1 topics")
        clusterer.name_topics(
            progress_callback=progress_callback,
            only_unnamed=only_unnamed,
            rebuild=rebuild,
            target='l1'
        )

    def run_stage9(self, only_unnamed: bool = True, rebuild: bool = False):
        """Run stage9: name L2 topics using LLM."""
        from src.ai.clustering import TopicClusterer
        
        # LLM client needed for naming
        llm_client = self._get_llm_client()
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=llm_client
        )
        
        # Create progress callback for topic naming
        def progress_callback(current, total, stage, total_all=None):
            percentage = int((current / total * 100)) if total > 0 else 0
            if total_all is not None and total_all != total:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%) (filtered/total: {total}/{total_all})", flush=True, end="")
            elif total_all is not None:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%) (all: {total_all})", flush=True, end="")
            else:
                print(f"\rNaming {stage.upper()} topics: {current}/{total} ({percentage}%)", flush=True, end="")
            if current == total:
                print()  # Newline after progress
        
        syslog2(LOG_NOTICE, "naming l2 topics")
        clusterer.name_topics(
            progress_callback=progress_callback,
            only_unnamed=only_unnamed,
            rebuild=rebuild,
            target='l2'
        )

    def run_stage6(self, **clustering_params):
        """Run stage6: L2 clustering - cluster L1 topics into L2 topics.
        Reads L1 centroids from SQLite, saves center_vec_json to SQLite, does NOT write to vector_db."""
        from src.ai.clustering import TopicClusterer
        
        # LLM client not needed for clustering - only for naming (stage9)
        clusterer = TopicClusterer(
            db=self.db,
            vector_store=self.vector_store,
            llm_client=None
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

    def run_stage7(self):
        """Run stage7: sync L2 topics from SQLite (center_vec_json) to vector_db."""
        syslog2(LOG_NOTICE, "starting stage7: syncing l2 topics to vector database")
        
        # Get all L2 topics with center_vec_json from SQLite
        l2_topics = self.db.get_all_topics_l2()
        
        if not l2_topics:
            syslog2(LOG_NOTICE, "no l2 topics found in sqlite")
            return
        
        # Prepare data for vector store
        ids = []
        embeddings = []
        metadatas = []
        
        for topic in l2_topics:
            if not topic.center_vec_json:
                syslog2(LOG_WARNING, "l2 topic missing center_vec_json", topic_id=topic.id)
                continue
            
            # Parse center_vec from JSON
            try:
                center_vec = json.loads(topic.center_vec_json)
            except (json.JSONDecodeError, TypeError) as e:
                syslog2(LOG_WARNING, "failed to parse center_vec_json for l2 topic", topic_id=topic.id, error=str(e))
                continue
            
            l2_topic_id = f"l2-{topic.id}"
            ids.append(l2_topic_id)
            embeddings.append(center_vec)
            metadatas.append({
                "topic_l2_id": topic.id,
                "title": topic.title or "unknown",
                "chunk_count": topic.chunk_count
            })
        
        if not ids:
            syslog2(LOG_WARNING, "no valid l2 topic centroids found to sync")
            return
        
        # Sync to vector store (add or update)
        # ChromaDB will update existing IDs automatically
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.vector_store.topics_l2_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            if i % (batch_size * 10) == 0:
                print(f"\rSyncing L2 topics: {min(i + batch_size, len(ids))}/{len(ids)}", flush=True, end="")
        
        print()  # Newline after progress
        syslog2(LOG_NOTICE, "l2 topics synced to vector database", count=len(ids))
        syslog2(LOG_NOTICE, "stage7 complete")

    def run_all(self, file_path: str, model: Optional[str] = None, batch_size: int = 128, **clustering_params):
        """Run all stages in sequence."""
        syslog2(LOG_NOTICE, "running stage0: parse and store messages")
        self.run_stage0(file_path)
        
        syslog2(LOG_NOTICE, "running stage1: create and store chunks")
        self.run_stage1()
        
        syslog2(LOG_NOTICE, "running stage2: generate embeddings for chunks (save to SQLite)")
        self.run_stage2(model=model, batch_size=batch_size)
        
        syslog2(LOG_NOTICE, "running stage3: sync chunks to vector database")
        self.run_stage3()
        
        syslog2(LOG_NOTICE, "running stage4: L1 clustering (save to SQLite)")
        self.run_stage4(**clustering_params)
        
        syslog2(LOG_NOTICE, "running stage5: sync L1 topics to vector database")
        self.run_stage5()
        
        syslog2(LOG_NOTICE, "running stage6: L2 clustering (save to SQLite)")
        self.run_stage6(**clustering_params)
        
        syslog2(LOG_NOTICE, "running stage7: sync L2 topics to vector database")
        self.run_stage7()
        
        syslog2(LOG_NOTICE, "running stage8: name L1 topics")
        self.run_stage8()
        
        syslog2(LOG_NOTICE, "running stage9: name L2 topics")
        self.run_stage9()
        
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

        # Determine chat_id from filename or default
        filename = os.path.basename(file_path)
        chat_id_match = re.search(r"telegram_dump_(-?\d+)", filename)
        chat_id = chat_id_match.group(1) if chat_id_match else "unknown_chat"
        syslog2(LOG_NOTICE, "identified chat_id", chat_id=chat_id)

        # Store messages in sql db
        syslog2(LOG_NOTICE, "preparing messages for database")
        db_messages = []
        for i, msg in enumerate(messages, 1):
            print(f"\rProcessing messages: {i}/{len(messages)}", flush=True, end="")
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
        print()  # Newline after progress
        syslog2(LOG_NOTICE, "messages prepared for database", count=len(db_messages))
        
        syslog2(LOG_NOTICE, "saving messages to database")
        try:
            inserted_count = self.db.add_messages_batch(db_messages)
            skipped_count = len(db_messages) - inserted_count
            if skipped_count > 0:
                syslog2(LOG_NOTICE, "messages saved to sql database", inserted=inserted_count, skipped=skipped_count, total=len(db_messages))
            else:
                syslog2(LOG_NOTICE, "messages saved to sql database", inserted=inserted_count, total=len(db_messages))
        except Exception as e:
            syslog2(LOG_ERR, "error saving messages", error=str(e))
            raise

        syslog2(LOG_NOTICE, "stage0 complete", messages_saved=inserted_count)

    def parse_and_store_chunks(self):
        """
        Create chunks from messages and store in SQLite database.
        Uses token-based chunking with parameters from profile config.
        
        Args:
        """
        syslog2(LOG_NOTICE, "starting parse and store chunks")

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

        # Use existing chunker from __init__ (already initialized with config parameters)
        # Create chunks
        syslog2(LOG_NOTICE, "creating chunks", 
                chunk_token_min=self.chunker.chunk_token_min, 
                chunk_token_max=self.chunker.chunk_token_max, 
                chunk_overlap_ratio=self.chunker.chunk_overlap_ratio)
        chunks = self.chunker.chunk_messages(messages)
        syslog2(LOG_NOTICE, "chunks created", count=len(chunks))

        # Store chunks in sql db
        chunk_models = []
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        session = self.db.get_session()
        try:
            syslog2(LOG_NOTICE, "preparing chunks for database")
            for i, chunk in enumerate(chunks, 1):
                print(f"\rProcessing chunks: {i}/{len(chunks)}", end="", flush=True)
                chunk_id = str(uuid.uuid4())
                
                # Prepare metadata (enhanced fields)
                # Convert chunk metadata to dict for metadata_json field
                meta_dict = {
                    "message_count": chunk.metadata.message_count,
                    "start_date": chunk.metadata.ts_from.isoformat(),
                    "end_date": chunk.metadata.ts_to.isoformat()
                }

                # Construct composite FKs for backward compatibility
                msg_id_start = f"{chat_id}_{chunk.metadata.msg_id_start}"
                msg_id_end = f"{chat_id}_{chunk.metadata.msg_id_end}"
                
                # Store raw msg_id (without chat_id prefix) for easier access
                msg_id_start_raw = chunk.metadata.msg_id_start
                msg_id_end_raw = chunk.metadata.msg_id_end

                model = ChunkModel(
                    id=chunk_id,
                    text=chunk.text,
                    metadata_json=json.dumps(meta_dict),
                    chat_id=chat_id,
                    msg_id_start=msg_id_start,
                    msg_id_end=msg_id_end,
                    msg_id_start_raw=msg_id_start_raw,
                    msg_id_end_raw=msg_id_end_raw,
                    ts_from=chunk.metadata.ts_from,
                    ts_to=chunk.metadata.ts_to
                )
                chunk_models.append(model)

                ids.append(chunk_id)
                documents.append(chunk.text)
                metadatas.append(meta_dict)
            print() # Newline after progress
            syslog2(LOG_NOTICE, "chunks prepared for database", count=len(chunk_models))

            syslog2(LOG_NOTICE, "saving chunks to database")
            session.add_all(chunk_models)
            session.commit()
            syslog2(LOG_NOTICE, "chunks saved to sql database", count=len(chunk_models))
        except Exception as e:
            session.rollback()
            syslog2(LOG_ERR, "error saving chunks", error=str(e))
            raise
        finally:
            session.close()

        syslog2(LOG_NOTICE, "stage1 complete", chunks_saved=len(chunk_models))

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
        Generate embeddings for chunks without embeddings and save to SQLite (chunks.embedding_json).
        Streaming version - processes chunks in batches to avoid memory issues.
        Does NOT write to vector database (that's stage3).
        
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
        
        # Get chunks from database that don't have embeddings - STREAMING APPROACH
        session = self.db.get_session()
        try:
            # Count total chunks to embed first
            total_to_embed = session.query(ChunkModel).filter(
                (ChunkModel.embedding_json.is_(None)) | 
                (ChunkModel.embedding_dim.is_(None)) | 
                (ChunkModel.embedding_dim != new_dimension)
            ).count()
            
            if total_to_embed == 0:
                syslog2(LOG_NOTICE, "all chunks already have embeddings")
                return
            
            total_chunks = session.query(ChunkModel).count()
            syslog2(LOG_NOTICE, "chunks to embed", total=total_chunks, missing=total_to_embed)
            
            # Process chunks in batches using yield_per - STREAMING APPROACH
            # This avoids loading all chunks into memory at once
            processed = 0
            query = session.query(ChunkModel).filter(
                (ChunkModel.embedding_json.is_(None)) | 
                (ChunkModel.embedding_dim.is_(None)) | 
                (ChunkModel.embedding_dim != new_dimension)
            )
            
            # Accumulate chunks in a batch
            batch_chunks = []
            for chunk in query.yield_per(batch_size):
                batch_chunks.append(chunk)
                
                # When batch is full, process it immediately
                if len(batch_chunks) >= batch_size:
                    # Prepare batch data
                    batch_ids = [chunk.id for chunk in batch_chunks]
                    batch_texts = [chunk.text for chunk in batch_chunks]
                    
                    # Generate embeddings for this batch only (not accumulating all)
                    batch_embeddings = emb_client.get_embeddings(batch_texts)
                    
                    # Save immediately to DB to free memory
                    for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                        embedding_json = json.dumps(embedding)
                        session.query(ChunkModel).filter(
                            ChunkModel.id == chunk_id
                        ).update({
                            ChunkModel.embedding_json: embedding_json,
                            ChunkModel.embedding_dim: new_dimension
                        }, synchronize_session=False)
                    
                    session.commit()
                    processed += len(batch_chunks)
                    
                    # Progress output
                    pct = (processed * 100) // total_to_embed if total_to_embed > 0 else 0
                    print(f"\rProcessing embeddings: {processed}/{total_to_embed} ({pct}%)", flush=True, end="")
                    
                    # Clear batch to free memory before next iteration
                    batch_chunks = []
            
            # Process remaining chunks (last incomplete batch)
            if batch_chunks:
                batch_ids = [chunk.id for chunk in batch_chunks]
                batch_texts = [chunk.text for chunk in batch_chunks]
                batch_embeddings = emb_client.get_embeddings(batch_texts)
                
                for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                    embedding_json = json.dumps(embedding)
                    session.query(ChunkModel).filter(
                        ChunkModel.id == chunk_id
                    ).update({
                        ChunkModel.embedding_json: embedding_json,
                        ChunkModel.embedding_dim: new_dimension
                    }, synchronize_session=False)
                
                session.commit()
                processed += len(batch_chunks)
            
            print()  # Newline after progress
            syslog2(LOG_NOTICE, "embeddings saved to sqlite database", count=processed)
            
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

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken (helper for ingest info)."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: approximate token count (1 token ≈ 4 characters)
            return len(text) // 4

    def get_ingest_info(self) -> str:
        """
        Get comprehensive ingestion information in compact format.
        Shows all stages, discrepancies, and profile/model info.
        
        Returns:
            Formatted string with ingest information
        """
        lines = []
        
        # Profile & Configuration Info
        profile_dir_str = str(self.profile_dir) if self.profile_dir else "unknown"
        lines.append("ingest info:")
        lines.append(f"profile_dir={profile_dir_str}")
        lines.append(f"db_url={self.db_url}")
        lines.append(f"vec_path={self.vector_store.persist_directory}")
        lines.append(f"collection={self.vector_store.collection_name}")
        lines.append("")
        
        # Get embedding model info from config
        embedding_model = "unknown"
        embedding_generator = "unknown"
        if self.profile_dir and self.profile_dir.exists():
            try:
                from src.bot.config import BotConfig
                config = BotConfig(self.profile_dir)
                embedding_model = config.data.get("embedding_model", "unknown")
                embedding_generator = config.embedding_generator
            except Exception:
                pass
        
        # Stage 0: Messages
        session = self.db.get_session()
        try:
            from src.storage.db import MessageModel
            all_messages = session.query(MessageModel).order_by(MessageModel.ts).all()
            
            total_messages = len(all_messages)
            
            # Group by chat_id
            chats = {}
            for msg in all_messages:
                chat_id = msg.chat_id or "unknown_chat"
                if chat_id not in chats:
                    chats[chat_id] = []
                chats[chat_id].append(msg)
            
            lines.append("stage0 messages:")
            lines.append(f"total={total_messages}")
            lines.append(f"chats={len(chats)}")
            lines.append("per_chat:")
            
            for chat_id, msgs in sorted(chats.items()):
                first_ts = msgs[0].ts.strftime("%Y-%m-%d %H:%M") if msgs else "unknown"
                last_ts = msgs[-1].ts.strftime("%Y-%m-%d %H:%M") if msgs else "unknown"
                lines.append(f"{chat_id}: messages={len(msgs)} first={first_ts} last={last_ts}")
            
            if not chats:
                lines.append("(no messages)")
        except Exception as e:
            lines.append("stage0 messages:")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 1: Chunks
        session = self.db.get_session()
        try:
            all_chunks = session.query(ChunkModel).all()
            total_chunks = len(all_chunks)
            
            # Group by chat_id
            chunks_by_chat = {}
            total_tokens = 0
            small_chunks = 0
            chunks_without_messages = 0
            
            # Get chunk_token_min from config for small chunk detection
            chunk_token_min = 50  # default
            if self.profile_dir and self.profile_dir.exists():
                try:
                    from src.bot.config import BotConfig
                    config = BotConfig(self.profile_dir)
                    chunk_token_min = config.chunk_token_min
                except Exception:
                    pass
            
            for chunk in all_chunks:
                chat_id = chunk.chat_id or "unknown_chat"
                if chat_id not in chunks_by_chat:
                    chunks_by_chat[chat_id] = 0
                chunks_by_chat[chat_id] += 1
                
                # Count tokens
                tokens = self._count_tokens(chunk.text)
                total_tokens += tokens
                
                if tokens < chunk_token_min:
                    small_chunks += 1
                
                # Check if chunk has messages
                if not chunk.msg_id_start:
                    chunks_without_messages += 1
            
            avg_tokens = total_tokens // total_chunks if total_chunks > 0 else 0
            
            lines.append("stage1 chunks:")
            lines.append(f"total={total_chunks}")
            lines.append("per_chat:")
            
            for chat_id, count in sorted(chunks_by_chat.items()):
                lines.append(f"{chat_id}: chunks={count}")
            
            if not chunks_by_chat:
                lines.append("(no chunks)")
            
            lines.append(f"avg_tokens_per_chunk={avg_tokens}")
            lines.append(f"small_chunks_below_min={small_chunks}")
            lines.append(f"chunks_without_messages={chunks_without_messages}")
        except Exception as e:
            lines.append("stage1 chunks:")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 2: Embeddings (SQLite)
        session = self.db.get_session()
        try:
            total_chunks = session.query(ChunkModel).count()
            chunks_with_embeddings_sql = session.query(ChunkModel).filter(
                ChunkModel.embedding_json.isnot(None)
            ).count()
            chunks_without_embeddings_sql = total_chunks - chunks_with_embeddings_sql
            
            lines.append("stage2 embeddings (sqlite):")
            lines.append(f"chunks_with_embeddings={chunks_with_embeddings_sql}")
            lines.append(f"chunks_without_embeddings={chunks_without_embeddings_sql}")
            lines.append(f"embedding_model={embedding_model}")
            lines.append(f"embedding_generator={embedding_generator}")
        except Exception as e:
            lines.append("stage2 embeddings (sqlite):")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 3: Vector DB Chunks
        try:
            vector_data = self.vector_store.get_all_embeddings()
            vector_ids = list(vector_data.get("ids", [])) if vector_data.get("ids") is not None else []
            vectors_total = len(vector_ids)
            
            # Get vector dimension
            vector_dim = 0
            embeddings = vector_data.get("embeddings", [])
            if embeddings is not None and len(embeddings) > 0:
                first_emb = embeddings[0]
                if hasattr(first_emb, '__len__'):
                    vector_dim = len(first_emb)
            
            session = self.db.get_session()
            try:
                total_chunks = session.query(ChunkModel).count()
                if vectors_total > 0:
                    vector_ids_set = set(vector_ids)
                    sql_chunk_ids = set(chunk.id for chunk in session.query(ChunkModel).all())
                    extra_vectors_without_chunks = len(vector_ids_set - sql_chunk_ids)
                else:
                    extra_vectors_without_chunks = 0
            finally:
                session.close()
            
            lines.append("stage3 vector_db chunks:")
            lines.append(f"vectors_total={vectors_total}")
            lines.append(f"vector_dim={vector_dim}")
            lines.append(f"extra_vectors_without_chunks={extra_vectors_without_chunks}")
        except Exception as e:
            lines.append("stage3 vector_db chunks:")
            lines.append(f"error: {str(e)}")
        
        lines.append("")
        
        # Stage 4: Topics L1 (SQLite)
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL1Model
            l1_topics = session.query(TopicL1Model).all()
            topics_l1_count = len(l1_topics)
            
            # Count chunks with topic_l1_id
            chunks_with_topic_l1 = session.query(ChunkModel).filter(ChunkModel.topic_l1_id.isnot(None)).count()
            total_chunks = session.query(ChunkModel).count()
            chunks_without_topic_l1 = total_chunks - chunks_with_topic_l1
            
            # Count topics with center_vec_json
            topics_with_centroids = session.query(TopicL1Model).filter(
                TopicL1Model.center_vec_json.isnot(None)
            ).count()
            
            lines.append("stage4 topics_l1 (sqlite):")
            lines.append(f"topics={topics_l1_count}")
            lines.append(f"topics_with_centroids={topics_with_centroids}")
            lines.append(f"chunks_with_topic_l1={chunks_with_topic_l1}")
            lines.append(f"chunks_without_topic_l1={chunks_without_topic_l1}")
        except Exception as e:
            lines.append("stage4 topics_l1 (sqlite):")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 5: Vector DB Topics L1
        try:
            l1_vector_data = self.vector_store.topics_l1_collection.get()
            l1_vectors_total = len(l1_vector_data.get("ids", [])) if l1_vector_data.get("ids") else 0
            
            lines.append("stage5 vector_db topics_l1:")
            lines.append(f"vectors_total={l1_vectors_total}")
        except Exception as e:
            lines.append("stage5 vector_db topics_l1:")
            lines.append(f"error: {str(e)}")
        
        lines.append("")
        
        # Stage 6: Topics L2 (SQLite)
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL2Model, TopicL1Model
            l2_topics = session.query(TopicL2Model).all()
            topics_l2_count = len(l2_topics)
            
            # Count L1 topics with parent_l2_id
            l1_with_parent = session.query(TopicL1Model).filter(TopicL1Model.parent_l2_id.isnot(None)).count()
            l1_without_parent = session.query(TopicL1Model).filter(TopicL1Model.parent_l2_id.is_(None)).count()
            
            # Count topics with center_vec_json
            topics_with_centroids = session.query(TopicL2Model).filter(
                TopicL2Model.center_vec_json.isnot(None)
            ).count()
            
            lines.append("stage6 topics_l2 (sqlite):")
            lines.append(f"topics={topics_l2_count}")
            lines.append(f"topics_with_centroids={topics_with_centroids}")
            lines.append(f"l1_with_parent_l2={l1_with_parent}")
            lines.append(f"l1_without_parent_l2={l1_without_parent}")
        except Exception as e:
            lines.append("stage6 topics_l2 (sqlite):")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 7: Vector DB Topics L2
        try:
            l2_vector_data = self.vector_store.topics_l2_collection.get()
            l2_vectors_total = len(l2_vector_data.get("ids", [])) if l2_vector_data.get("ids") else 0
            
            lines.append("stage7 vector_db topics_l2:")
            lines.append(f"vectors_total={l2_vectors_total}")
        except Exception as e:
            lines.append("stage7 vector_db topics_l2:")
            lines.append(f"error: {str(e)}")
        
        lines.append("")
        
        # Stage 8: L1 Topic Names
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL1Model
            l1_topics = session.query(TopicL1Model).all()
            named_l1 = sum(1 for t in l1_topics if t.title and t.title.lower() not in ("unknown", "") and not t.title.startswith("Topic L1-"))
            unnamed_l1 = len(l1_topics) - named_l1
            
            lines.append("stage8 l1 topic names:")
            lines.append(f"named={named_l1}")
            lines.append(f"unnamed={unnamed_l1}")
        except Exception as e:
            lines.append("stage8 l1 topic names:")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        lines.append("")
        
        # Stage 9: L2 Topic Names
        session = self.db.get_session()
        try:
            from src.storage.db import TopicL2Model
            l2_topics = session.query(TopicL2Model).all()
            named_l2 = sum(1 for t in l2_topics if t.title and t.title.lower() not in ("unknown", "") and not t.title.startswith("Topic L2-"))
            unnamed_l2 = len(l2_topics) - named_l2
            
            lines.append("stage9 l2 topic names:")
            lines.append(f"named={named_l2}")
            lines.append(f"unnamed={unnamed_l2}")
        except Exception as e:
            lines.append("stage9 l2 topic names:")
            lines.append(f"error: {str(e)}")
        finally:
            session.close()
        
        return "\n".join(lines)


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
        collection = parse_option(stream, "collection") or "embed-l1"
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