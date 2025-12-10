import numpy as np
import hdbscan
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import random
import warnings
import time

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.core.syslog2 import *
from src.core.llm import LLMClient
from src.core.prompt import *

# Suppress sklearn deprecation warnings from hdbscan
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

MAX_L1_TOPICS_FOR_L2_NAMING = 5
MAX_REPRESENTATIVE_CHUNKS_PER_L1 = 5
MAX_TOTAL_CHUNKS_FOR_L2_NAMING = 25
MAX_L2_CHUNK_TEXT_LEN = 4096
MIN_CHUNK_TEXT_LEN = 30


class TopicClusterer:
    """
    Handles hierarchical topic clustering logic.
    Phase 14.3: L1 Clustering (Fine-grained)
    Phase 14.4: L2 Clustering (Super-topics)
    Phase 14.5: Topic Naming (LLM)
    """
    def __init__(self, db: Database, vector_store: VectorStore, llm_client: Optional[LLMClient] = None):
        self.db = db
        self.vector_store = vector_store
        self.llm_client = llm_client
    
    def _show_progress(self, current: int, total: int, message: str) -> None:
        """
        Show progress bar.
        
        Args:
            current: Current progress
            total: Total items
            message: Progress message
        """
        percentage = int((current / total * 100)) if total > 0 else 0
        print(f"\r{message}: {current}/{total} ({percentage}%)", flush=True, end="")
    
    def _update_topic_metadata(self, topic_id: int, topic_type: str, parent_id: Optional[int] = None) -> None:
        """
        Update topic metadata (parent relationship).
        
        Args:
            topic_id: Topic ID to update
            topic_type: "l1" or "l2"
            parent_id: Parent topic ID (None to clear)
        """
        if topic_type == "l1":
            self.db.update_topic_l1_parent(topic_id, parent_l2_id=parent_id)
        # L2 topics don't have parents in current implementation
    
    def _update_chunk_metadata_batch(
        self, 
        chunk_ids: List[str], 
        topic_l1_id: Optional[int], 
        topic_l2_id: Optional[int]
    ) -> None:
        """
        Batch update chunk metadata in SQLite and ChromaDB.
        
        Args:
            chunk_ids: List of chunk IDs to update
            topic_l1_id: L1 topic ID (None to clear)
            topic_l2_id: L2 topic ID (None to clear)
        """
        for chunk_id in chunk_ids:
            # Update SQLite
            self.db.update_chunk_topics(chunk_id, topic_l1_id=topic_l1_id, topic_l2_id=topic_l2_id)
            
            # Update ChromaDB metadata
            try:
                existing = self.vector_store.get_embeddings_by_ids([chunk_id])
                existing_meta = existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
                if topic_l1_id is not None:
                    existing_meta["topic_l1_id"] = topic_l1_id
                else:
                    existing_meta.pop("topic_l1_id", None)
                if topic_l2_id is not None:
                    existing_meta["topic_l2_id"] = topic_l2_id
                else:
                    existing_meta.pop("topic_l2_id", None)
                self.vector_store.update_chunk_metadata(chunk_id, existing_meta)
            except Exception as e:
                syslog2(LOG_DEBUG, "failed to update chunk metadata in chroma", chunk_id=chunk_id, error=str(e))

    def perform_l1_clustering(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        metric: str = 'cosine',
        cluster_selection_method: str = 'eom',
        cluster_selection_epsilon: float = 0.0
    ):
        """
        Fetches all chunks, clusters them using HDBSCAN, and saves L1 topics.

        Args:
            min_cluster_size: Minimum number of chunks in a cluster (default: 2)
            min_samples: Minimum samples in neighborhood (default: 1)
            metric: Distance metric ('cosine', 'euclidean', 'manhattan', etc.)
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf' (Leaf)
            cluster_selection_epsilon: A distance threshold (0.0 = automatic)
        """
        syslog2(LOG_NOTICE, "starting l1 clustering",
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                method=cluster_selection_method)

        # 1. fetch data from sqlite (chunks.embedding_json) – source of truth
        session = self.db.get_session()
        try:
            from src.storage.db import ChunkModel
            chunks_with_embeddings = session.query(ChunkModel).filter(
                ChunkModel.embedding_json.isnot(None)
            ).all()
            
            if not chunks_with_embeddings:
                syslog2(LOG_DEBUG, "no embeddings in sqlite, trying vector_db")
                data = self.vector_store.get_all_embeddings()
                ids = data.get('ids', [])
                embeddings = data.get('embeddings', [])
                metadatas = data.get('metadatas', [])
                
                if not ids or len(embeddings) == 0:
                    syslog2(LOG_WARNING, "no data for clustering")
                    return
            else:
                ids = []
                embeddings = []
                metadatas = []
                
                for chunk in chunks_with_embeddings:
                    try:
                        embedding = json.loads(chunk.embedding_json)
                        ids.append(chunk.id)
                        embeddings.append(embedding)
                        
                        meta_dict = {}
                        if chunk.metadata_json:
                            try:
                                meta_dict = json.loads(chunk.metadata_json)
                            except Exception as e:
                                syslog2(LOG_DEBUG, "metadata_json parse failed", chunk_id=chunk.id, error=str(e))
                        if chunk.topic_l1_id is not None:
                            meta_dict["topic_l1_id"] = chunk.topic_l1_id
                        if chunk.topic_l2_id is not None:
                            meta_dict["topic_l2_id"] = chunk.topic_l2_id
                        metadatas.append(meta_dict)
                    except (json.JSONDecodeError, TypeError) as e:
                        syslog2(LOG_WARNING, "failed to parse embedding_json for chunk", chunk_id=chunk.id, error=str(e))
                        continue
                
                if not ids:
                    syslog2(LOG_WARNING, "no valid embeddings found in sqlite")
                    return
        finally:
            session.close()

        if not ids or embeddings is None or (isinstance(embeddings, list) and not embeddings) or (hasattr(embeddings, 'size') and embeddings.size == 0):
            syslog2(LOG_WARNING, "no data for clustering")
            return

        X = np.array(embeddings)

        if len(X.shape) != 2:
            syslog2(LOG_ERR, "invalid embeddings shape", shape=str(X.shape))
            return

        n_samples = len(ids)
        syslog2(LOG_NOTICE, "clustering chunks", count=n_samples)

        if min_cluster_size < 2:
            syslog2(LOG_ERR, "min_cluster_size must be at least 2 (hdbscan requirement)",
                    provided=min_cluster_size)
            raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")

        if min_cluster_size > n_samples:
            syslog2(LOG_WARNING, "min_cluster_size too large, adjusting",
                    original=min_cluster_size, adjusted=n_samples)
            min_cluster_size = max(2, n_samples // 2)

        actual_metric = metric
        if metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
            actual_metric = 'euclidean'
            syslog2(LOG_DEBUG, "normalized embeddings for cosine similarity, using euclidean metric")

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=actual_metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            labels = clusterer.fit_predict(X)
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "Min cluster size must be greater than one" in error_msg:
                syslog2(LOG_ERR, "hdbscan requires min_cluster_size >= 2", provided=min_cluster_size)
                raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}") from None
            elif "metric" in error_msg.lower() or "unrecognized" in error_msg.lower():
                raise ValueError(f"metric '{actual_metric}' not supported by hdbscan: {error_msg}") from None
            else:
                raise

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        noise_count = len(clusters.get(-1, []))
        valid_clusters = len(clusters) - (1 if -1 in clusters else 0)
        noise_percentage = (noise_count / n_samples * 100) if n_samples > 0 else 0.0

        syslog2(LOG_NOTICE, "clustering complete",
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")

        if noise_percentage > 50.0:
            syslog2(LOG_WARNING, "high noise percentage, consider lowering min_cluster_size or min_samples",
                    noise_percentage=f"{noise_percentage:.1f}%")

        # 4. save topics
        self.db.clear_topics_l1()
        self._l1_topic_assignments = {}

        total_clusters = len(clusters)
        processed = 0
        try:
            for label, indices in clusters.items():
                if label == -1:
                    # noise
                    for idx in indices:
                        chunk_id = ids[idx]
                        if None not in self._l1_topic_assignments:
                            self._l1_topic_assignments[None] = []
                        self._l1_topic_assignments[None].append(chunk_id)
                    processed += 1
                    self._show_progress(processed, total_clusters, "Creating L1 topics")
                    continue

                # нормальный кластер
                cluster_ids = [ids[i] for i in indices]
                cluster_embeddings = X[indices]
                cluster_metas = [metadatas[i] for i in indices]

                centroid = np.mean(cluster_embeddings, axis=0).tolist()
                chunk_count = len(cluster_ids)

                total_msg_count = 0
                ts_start_list = []
                ts_end_list = []

                for meta in cluster_metas:
                    if meta:
                        total_msg_count += int(meta.get('message_count', 0))
                        s_date = meta.get('start_date')
                        e_date = meta.get('end_date')
                        if s_date:
                            ts_start_list.append(s_date)
                        if e_date:
                            ts_end_list.append(e_date)

                ts_from = None
                ts_to = None

                if ts_start_list:
                    try:
                        ts_from = datetime.fromisoformat(min(ts_start_list))
                    except ValueError:
                        pass

                if ts_end_list:
                    try:
                        ts_to = datetime.fromisoformat(max(ts_end_list))
                    except ValueError:
                        pass

                topic_id = self.db.create_topic_l1(
                    title="unknown",
                    descr="Pending description...",
                    chunk_count=chunk_count,
                    msg_count=total_msg_count,
                    center_vec=centroid,
                    ts_from=ts_from,
                    ts_to=ts_to,
                    vector_store=None  # to vector_db in stage5
                )

                self._l1_topic_assignments[topic_id] = cluster_ids
                processed += 1
                self._show_progress(processed, total_clusters, "Creating L1 topics")
        finally:
            print()

        syslog2(LOG_NOTICE, "l1 topics created", count=valid_clusters)

        return self._l1_topic_assignments

    def assign_l1_topics_to_chunks(self, show_progress: bool = True):
        """
        Assign topic_l1_id to chunks based on clustering results.
        This should be called after perform_l1_clustering.
        """
        if not hasattr(self, '_l1_topic_assignments') or not self._l1_topic_assignments:
            syslog2(LOG_WARNING, "no l1 topic assignments found, run perform_l1_clustering first")
            return

        assigned_count = 0
        noise_count = 0

        total_chunks = sum(len(chunk_ids) for chunk_ids in self._l1_topic_assignments.values())

        processed = 0
        try:
            for topic_id, chunk_ids in self._l1_topic_assignments.items():
                if topic_id is None:
                    self._update_chunk_metadata_batch(chunk_ids, topic_l1_id=None, topic_l2_id=None)
                    noise_count += len(chunk_ids)
                    processed += len(chunk_ids)
                else:
                    self._update_chunk_metadata_batch(chunk_ids, topic_l1_id=topic_id, topic_l2_id=None)
                    assigned_count += len(chunk_ids)
                    processed += len(chunk_ids)
                
                if show_progress:
                    self._show_progress(
                        processed, 
                        total_chunks, 
                        f"Assigning topics to chunks (assigned: {assigned_count}, noise: {noise_count})"
                    )
        finally:
            if show_progress:
                print()  # Newline after progress

        syslog2(LOG_NOTICE, "l1 topics assigned to chunks", assigned=assigned_count, noise=noise_count)
        delattr(self, '_l1_topic_assignments')

    def perform_l2_clustering(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 1,
        metric: str = 'cosine',
        cluster_selection_method: str = 'eom',
        cluster_selection_epsilon: float = 0.0
    ):
        """
        Fetches L1 topics, clusters their centroids to form L2 topics.
        """
        syslog2(LOG_NOTICE, "starting l2 clustering")

        # 1. Fetch L1 topics
        l1_topics = self.db.get_all_topics_l1()
        if not l1_topics:
            syslog2(LOG_WARNING, "no l1 topics found for l2 clustering")
            return

        # Get L1 topic centroids from SQLite (topics_l1.center_vec_json) - source of truth
        # Fallback to ChromaDB if SQLite doesn't have centroids yet
        embeddings = []
        ids = []
        
        # Try SQLite first
        topics_with_centroids = [t for t in l1_topics if t.center_vec_json]
        
        if topics_with_centroids:
            # Read from SQLite
            for topic in topics_with_centroids:
                try:
                    center_vec = json.loads(topic.center_vec_json)
                    embeddings.append(center_vec)
                    ids.append(topic.id)
                except (json.JSONDecodeError, TypeError) as e:
                    syslog2(LOG_WARNING, "failed to parse center_vec_json for l1 topic", topic_id=topic.id, error=str(e))
                    continue
        else:
            # Fallback to ChromaDB
            syslog2(LOG_DEBUG, "no centroids in sqlite, trying chroma_db")
            try:
                l1_data = self.vector_store.topics_l1_collection.get(
                    include=["embeddings", "metadatas"]
                )
                
                if not l1_data or not l1_data.get("ids") or not l1_data["ids"]:
                    syslog2(LOG_WARNING, "no l1 topic centroids found in chroma_db")
                    return
                
                chroma_ids = l1_data["ids"]
                chroma_embeddings = l1_data.get("embeddings", [])
                
                # Build mapping and collect embeddings
                for idx, chroma_id in enumerate(chroma_ids):
                    try:
                        # Extract topic ID from "l1-123" format
                        topic_id = int(chroma_id.replace("l1-", ""))
                        
                        # Verify topic exists in database
                        if topic_id not in [t.id for t in l1_topics]:
                            continue
                        
                        if idx < len(chroma_embeddings):
                            embeddings.append(chroma_embeddings[idx])
                            ids.append(topic_id)
                    except (ValueError, IndexError):
                        continue
                        
            except Exception as e:
                syslog2(LOG_WARNING, "failed to get l1 topic centroids from chroma_db", error=str(e))
                return
        
        if not embeddings:
            syslog2(LOG_WARNING, "no valid l1 topic centroids found")
            return

        X = np.array(embeddings)
        n_l1_topics = len(X)

        if min_cluster_size < 2:
            syslog2(LOG_ERR, "min_cluster_size must be at least 2 (HDBSCAN requirement)",
                    provided=min_cluster_size)
            raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")

        if n_l1_topics < min_cluster_size:
            syslog2(LOG_NOTICE, "not enough l1 topics for clustering",
                    count=n_l1_topics, min_required=min_cluster_size)
            return

        syslog2(LOG_NOTICE, "clustering l1 topics", count=n_l1_topics)

        actual_metric = metric
        if metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
            actual_metric = 'euclidean'
            syslog2(LOG_DEBUG, "normalized embeddings for cosine similarity, using euclidean metric")

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=actual_metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            labels = clusterer.fit_predict(X)
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "Min cluster size must be greater than one" in error_msg:
                syslog2(LOG_ERR, "HDBSCAN requires min_cluster_size >= 2", provided=min_cluster_size)
                raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}") from None
            elif "metric" in error_msg.lower() or "unrecognized" in error_msg.lower():
                raise ValueError(f"Metric '{actual_metric}' not supported by HDBSCAN: {error_msg}") from None
            else:
                raise

        self.db.clear_topics_l2()

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        noise_count = len(clusters.get(-1, []))
        valid_clusters = len(clusters) - (1 if -1 in clusters else 0)
        noise_percentage = (noise_count / n_l1_topics * 100) if n_l1_topics > 0 else 0

        syslog2(LOG_NOTICE, "l2 clustering complete",
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")

        processed = 0
        try:
            for label, indices in clusters.items():
                if label == -1:
                    # noise
                    for idx in indices:
                        l1_id = ids[idx]
                        self._update_topic_metadata(l1_id, "l1", parent_id=None)
                        self.db.update_chunks_parent_l2(l1_id, topic_l2_id=None)
                    processed += 1
                    self._show_progress(processed, len(clusters), "Creating L2 topics")
                    continue

                cluster_indices = indices
                cluster_l1_ids = [ids[i] for i in cluster_indices]
                cluster_embeddings = X[cluster_indices]

                centroid = np.mean(cluster_embeddings, axis=0).tolist()
                syslog2(LOG_DEBUG, "computed l2 centroid", 
                       cluster_label=label, 
                       centroid_dim=len(centroid),
                       l1_topics_count=len(cluster_l1_ids))

                total_chunk_count = 0
                for i in cluster_indices:
                    l1_topic = next(t for t in l1_topics if t.id == ids[i])
                    total_chunk_count += l1_topic.chunk_count

                syslog2(LOG_DEBUG, "creating l2 topic", 
                       cluster_label=label,
                       chunk_count=total_chunk_count,
                       centroid_dim=len(centroid))
                
                # Save topic to SQLite with center_vec_json (NOT to vector_db - that's stage7)
                l2_id = self.db.create_topic_l2(
                    title="unknown",
                    descr="Pending description...",
                    chunk_count=total_chunk_count,
                    center_vec=centroid,
                    vector_store=None  # Don't save to vector_db on stage6
                )
                
                syslog2(LOG_DEBUG, "l2 topic created and saved to chroma_db", 
                       l2_id=l2_id, 
                       l2_topic_id=f"l2-{l2_id}",
                       centroid_dim=len(centroid))

                for l1_id in cluster_l1_ids:
                    self._update_topic_metadata(l1_id, "l1", parent_id=l2_id)
                    # Update chunks in SQLite
                    updated_count = self.db.update_chunks_parent_l2(l1_id, topic_l2_id=l2_id)
                    # Update metadata in chroma_db for all affected chunks
                    if updated_count > 0:
                        try:
                            chunks = self.db.get_chunks_by_topic_l1(l1_id)
                            chunk_ids = [chunk.id for chunk in chunks]
                            # Use batch update helper
                            for chunk in chunks:
                                topic_l1_id = chunk.topic_l1_id if chunk.topic_l1_id else None
                                self._update_chunk_metadata_batch([chunk.id], topic_l1_id=topic_l1_id, topic_l2_id=l2_id)
                        except Exception as e:
                            syslog2(LOG_WARNING, "failed to update chunk metadata in chroma for l2 assignment", l1_id=l1_id, l2_id=l2_id, error=str(e))

                processed += 1
                self._show_progress(processed, len(clusters), "Creating L2 topics")
        finally:
            print()  # Newline after progress

        syslog2(LOG_NOTICE, "l2 topics saved")

    def name_topics(self, progress_callback=None, only_unnamed: bool = True, rebuild: bool = False, target: str = 'both'):
        """
        Generates names for L1 and L2 topics using LLM.
        """
        if not self.llm_client:
            syslog2(LOG_WARNING, "llm client not available for topic naming")
            return

        syslog2(LOG_WARNING, "starting topic naming", only_unnamed=only_unnamed, rebuild=rebuild, target=target)

        # L1
        if target in ('l1', 'both'):
            l1_topics_all = self.db.get_all_topics_l1()
            total_l1_all = len(l1_topics_all)

            if rebuild:
                l1_topics = l1_topics_all
            elif only_unnamed:
                l1_topics = [
                    t for t in l1_topics_all
                    if (not t.title
                        or t.title.startswith("Topic L1-")
                        or t.title.strip() == ""
                        or t.title.lower().strip() == "unknown")
                ]
            else:
                l1_topics = l1_topics_all

            total_l1_filtered = len(l1_topics)
            syslog2(LOG_NOTICE, "naming l1 topics", filtered=total_l1_filtered, total=total_l1_all)

            for idx, topic in enumerate(l1_topics, 1):
                self._name_l1_topic(topic)
                if progress_callback:
                    progress_callback(idx, total_l1_filtered, 'l1', total_all=total_l1_all)

        # L2
        if target in ('l2', 'both'):
            l2_topics_all = self.db.get_all_topics_l2()
            total_l2_all = len(l2_topics_all)

            if rebuild:
                l2_topics = l2_topics_all
            elif only_unnamed:
                l2_topics = [
                    t for t in l2_topics_all
                    if (not t.title
                        or t.title.strip() == ""
                        or t.title.lower().strip() == "unknown")
                ]
            else:
                l2_topics = l2_topics_all

            total_l2_filtered = len(l2_topics)
            syslog2(LOG_NOTICE, "naming l2 topics", filtered=total_l2_filtered, total=total_l2_all)

            for idx, topic in enumerate(l2_topics, 1):
                self._name_l2_topic(topic)
                if progress_callback:
                    progress_callback(idx, total_l2_filtered, 'l2', total_all=total_l2_all)

        syslog2(LOG_NOTICE, "topic naming complete")

    def _name_l1_topic(self, topic) -> None:
        """Helper to name a single L1 topic."""
        chunks = self.db.get_chunks_by_topic_l1(topic.id)
        if not chunks:
            return

        sample_size = min(len(chunks), 5)
        sample_chunks = random.sample(chunks, sample_size)

        messages_text = "\n\n".join(
            [f"Chunk {i + 1}:\n{c.text[:200]}..." for i, c in enumerate(sample_chunks)]
        )

        prompt = TOPIC_L1_NAMING_PROMPT.format(messages=messages_text)

        try:
            response_json = self._call_llm_json(prompt, max_retries=3)
            if response_json:
                title = response_json.get("title", f"Topic {topic.id}")
                description = response_json.get("description", "No description generated.")
                self.db.update_topic_l1_info(topic.id, title, description)
            else:
                syslog2(LOG_WARNING, "failed to name l1 topic after retries, setting to unknown", id=topic.id)
                self.db.update_topic_l1_info(topic.id, "unknown", "Failed to generate description after retries.")
        except Exception as e:
            syslog2(LOG_ERR, "failed to name l1 topic", id=topic.id, error=str(e))
            try:
                self.db.update_topic_l1_info(topic.id, "unknown", f"Error: {str(e)}")
            except Exception:
                pass

    def _name_l2_topic(self, topic) -> None:
        """Helper to name a single L2 topic using representative chunks."""
        syslog2(LOG_DEBUG, "naming l2 topic start", l2_id=topic.id)
        
        # Get l2 center from ChromaDB (not from SQLite)
        l2_topic_id = f"l2-{topic.id}"
        try:
            syslog2(LOG_DEBUG, "fetching l2 center from chroma_db", l2_id=topic.id, l2_topic_id=l2_topic_id)
            l2_data = self.vector_store.topics_l2_collection.get(
                ids=[l2_topic_id],
                include=["embeddings", "metadatas"]
            )
            
            syslog2(LOG_DEBUG, "l2_data received from chroma_db", 
                   l2_id=topic.id,
                   l2_data_type=type(l2_data).__name__,
                   l2_data_keys=list(l2_data.keys()) if isinstance(l2_data, dict) else "not a dict")
            
            # Check if data exists and has required fields
            # Use safe checks to avoid numpy array truth value errors
            if not isinstance(l2_data, dict):
                syslog2(LOG_WARNING, "l2_data is not a dict", l2_id=topic.id, l2_topic_id=l2_topic_id, data_type=type(l2_data).__name__)
                return
            
            ids_list = l2_data.get("ids")
            embeddings_list = l2_data.get("embeddings")
            
            # Safe length checks
            ids_count = len(ids_list) if ids_list is not None else 0
            embeddings_count = len(embeddings_list) if embeddings_list is not None else 0
            
            syslog2(LOG_DEBUG, "checking l2_data contents", 
                   l2_id=topic.id,
                   has_ids=ids_list is not None,
                   ids_count=ids_count,
                   has_embeddings=embeddings_list is not None,
                   embeddings_count=embeddings_count)
            
            if ids_count == 0 or embeddings_count == 0:
                syslog2(LOG_WARNING, "l2 topic embedding not found in chroma_db, skipping naming", 
                       id=topic.id, l2_topic_id=l2_topic_id,
                       ids_count=ids_count,
                       embeddings_count=embeddings_count)
                return
            
            # Get first embedding and convert to numpy array
            first_embedding = embeddings_list[0]
            l2_center_vec = np.array(first_embedding, dtype=float)
            syslog2(LOG_DEBUG, "retrieved l2 center from chroma_db", 
                   l2_id=topic.id, 
                   embedding_dim=len(l2_center_vec),
                   embedding_type=type(first_embedding).__name__)
            
        except Exception as e:
            syslog2(LOG_WARNING, "failed to get l2 center from chroma_db, skipping naming", 
                   id=topic.id, l2_topic_id=l2_topic_id, error=str(e))
            return

        subtopics = self.db.get_l1_topics_by_l2(topic.id)
        if not subtopics:
            syslog2(LOG_WARNING, "no l1 subtopics found for l2 topic, skipping naming", id=topic.id)
            return
        
        syslog2(LOG_DEBUG, "l2 topic has subtopics", l2_id=topic.id, subtopic_count=len(subtopics))

        # precompute l1 centers
        l1_centers: Dict[int, np.ndarray] = {}
        l1_scored: List[Tuple[float, object]] = []
        l1_without_center = 0
        l1_invalid_center = 0

        # Get L1 topic centroids from ChromaDB
        l1_chroma_ids = [f"l1-{t.id}" for t in subtopics]
        try:
            l1_data = self.vector_store.topics_l1_collection.get(
                ids=l1_chroma_ids,
                include=["embeddings", "metadatas"]
            )
            
            chroma_ids = l1_data.get("ids", [])
            chroma_embeddings = l1_data.get("embeddings", [])
            
            # Build mapping from ChromaDB IDs to topic IDs
            for idx, chroma_id in enumerate(chroma_ids):
                try:
                    topic_id = int(chroma_id.replace("l1-", ""))
                    if topic_id not in [t.id for t in subtopics]:
                        continue
                    
                    if idx < len(chroma_embeddings):
                        vec = np.array(chroma_embeddings[idx], dtype=float)
                        l1_centers[topic_id] = vec
                        # Find the topic object
                        topic_obj = next((t for t in subtopics if t.id == topic_id), None)
                        if topic_obj:
                            sim = self._cosine_similarity(l2_center_vec, vec)
                            l1_scored.append((sim, topic_obj))
                except (ValueError, IndexError, TypeError) as e:
                    l1_invalid_center += 1
                    syslog2(LOG_DEBUG, "l1 topic has invalid center_vec from chroma_db", l1_id=chroma_id, l2_id=topic.id, error=str(e))
            
            # Count topics without centers
            for t in subtopics:
                if t.id not in l1_centers:
                    l1_without_center += 1
                    
        except Exception as e:
            syslog2(LOG_WARNING, "failed to get l1 topic centroids from chroma_db for l2 naming", l2_id=topic.id, error=str(e))
            for t in subtopics:
                l1_without_center += 1

        if not l1_scored:
            syslog2(LOG_WARNING, "no valid l1 centers for l2 naming", 
                   id=topic.id, 
                   total_subtopics=len(subtopics),
                   without_center=l1_without_center,
                   invalid_center=l1_invalid_center)
            return

        syslog2(LOG_DEBUG, "l1 centers computed", 
               l2_id=topic.id, 
               valid_centers=len(l1_scored),
               without_center=l1_without_center,
               invalid_center=l1_invalid_center)

        l1_scored.sort(key=lambda x: x[0], reverse=True)
        top_l1_topics = [t for _, t in l1_scored[:MAX_L1_TOPICS_FOR_L2_NAMING]]
        syslog2(LOG_DEBUG, "selected top l1 topics for l2 naming", 
               l2_id=topic.id, 
               selected_count=len(top_l1_topics),
               max_allowed=MAX_L1_TOPICS_FOR_L2_NAMING)

        selected_chunks_texts: List[str] = []
        seen_texts: set = set()  # Track normalized texts to avoid duplicates
        total_limit = MAX_TOTAL_CHUNKS_FOR_L2_NAMING
        stats_no_chunks = 0
        stats_no_chunk_ids = 0
        stats_no_embeddings = 0
        stats_no_scored = 0
        stats_no_text = 0
        stats_too_short = 0
        stats_duplicate = 0

        for l1_topic in top_l1_topics:
            if len(selected_chunks_texts) >= total_limit:
                syslog2(LOG_DEBUG, "reached total limit for chunks", l2_id=topic.id, collected=len(selected_chunks_texts))
                break

            chunks = self.db.get_chunks_by_topic_l1(l1_topic.id)
            if not chunks:
                stats_no_chunks += 1
                syslog2(LOG_DEBUG, "no chunks found for l1 topic", l1_id=l1_topic.id, l2_id=topic.id)
                continue

            chunk_ids = [c.id for c in chunks if getattr(c, "id", None) is not None]
            if not chunk_ids:
                stats_no_chunk_ids += 1
                syslog2(LOG_DEBUG, "no valid chunk ids for l1 topic", l1_id=l1_topic.id, l2_id=topic.id, total_chunks=len(chunks))
                continue

            emb_map = self._get_chunk_embeddings(chunk_ids)
            if not emb_map:
                stats_no_embeddings += 1
                syslog2(LOG_WARNING, "failed to load embeddings for l1 topic chunks", 
                       l1_id=l1_topic.id, l2_id=topic.id, chunk_count=len(chunk_ids))
                continue

            l1_center = l1_centers.get(l1_topic.id)
            if l1_center is None:
                syslog2(LOG_DEBUG, "l1 center not found in precomputed centers", l1_id=l1_topic.id, l2_id=topic.id)
                continue

            scored_chunks: List[Tuple[float, object]] = []
            chunks_without_id = 0
            chunks_without_embedding = 0
            
            for c in chunks:
                cid = getattr(c, "id", None)
                if cid is None:
                    chunks_without_id += 1
                    continue
                vec = emb_map.get(cid)
                if vec is None:
                    chunks_without_embedding += 1
                    continue
                sim = self._cosine_similarity(l1_center, vec)
                scored_chunks.append((sim, c))

            if not scored_chunks:
                stats_no_scored += 1
                syslog2(LOG_DEBUG, "no scored chunks for l1 topic", 
                       l1_id=l1_topic.id, l2_id=topic.id,
                       total_chunks=len(chunks),
                       without_id=chunks_without_id,
                       without_embedding=chunks_without_embedding)
                continue

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            per_l1_limit = min(MAX_REPRESENTATIVE_CHUNKS_PER_L1,
                               total_limit - len(selected_chunks_texts))

            chunks_added_from_l1 = 0
            for sim, c in scored_chunks[:per_l1_limit]:
                text = c.text if hasattr(c, "text") and c.text else ""
                if not text:
                    stats_no_text += 1
                    syslog2(LOG_DEBUG, "chunk has no text, skipping", 
                           chunk_id=getattr(c, "id", None), l1_id=l1_topic.id, l2_id=topic.id)
                    continue
                
                # Check minimum length
                if len(text.strip()) < MIN_CHUNK_TEXT_LEN:
                    stats_too_short += 1
                    syslog2(LOG_DEBUG, "chunk text too short, skipping", 
                           chunk_id=getattr(c, "id", None), l1_id=l1_topic.id, l2_id=topic.id,
                           text_length=len(text.strip()), min_required=MIN_CHUNK_TEXT_LEN)
                    continue
                
                # Normalize text for duplicate detection (strip and lowercase)
                normalized_text = text.strip().lower()
                if normalized_text in seen_texts:
                    stats_duplicate += 1
                    syslog2(LOG_DEBUG, "chunk text duplicate, skipping", 
                           chunk_id=getattr(c, "id", None), l1_id=l1_topic.id, l2_id=topic.id)
                    continue
                
                # Add to seen texts and append to selected chunks
                seen_texts.add(normalized_text)
                trimmed = text[:MAX_L2_CHUNK_TEXT_LEN]
                block = f"--- chunk from topic l1-{l1_topic.id} ---\n{trimmed}"
                selected_chunks_texts.append(block)
                chunks_added_from_l1 += 1

                if len(selected_chunks_texts) >= total_limit:
                    break
            
            syslog2(LOG_DEBUG, "processed l1 topic for l2 naming", 
                   l1_id=l1_topic.id, l2_id=topic.id,
                   total_chunks=len(chunks),
                   scored_chunks=len(scored_chunks),
                   chunks_added=chunks_added_from_l1,
                   total_collected=len(selected_chunks_texts))

        if not selected_chunks_texts:
            syslog2(LOG_WARNING, "no representative chunks selected for l2 naming", 
                   id=topic.id,
                   top_l1_count=len(top_l1_topics),
                   stats_no_chunks=stats_no_chunks,
                   stats_no_chunk_ids=stats_no_chunk_ids,
                   stats_no_embeddings=stats_no_embeddings,
                   stats_no_scored=stats_no_scored,
                   stats_no_text=stats_no_text,
                   stats_too_short=stats_too_short,
                   stats_duplicate=stats_duplicate)
            return

        syslog2(LOG_DEBUG, "collected representative chunks for l2 naming", 
               l2_id=topic.id,
               chunks_count=len(selected_chunks_texts),
               total_text_length=sum(len(t) for t in selected_chunks_texts))

        chunks_text = "\n\n".join(selected_chunks_texts)
        prompt = TOPIC_L2_NAMING_PROMPT.format(subtopics=chunks_text)
        
        syslog2(LOG_DEBUG, "calling llm for l2 topic naming", l2_id=topic.id, prompt_length=len(prompt))

        try:
            response_json = self._call_llm_json(prompt, max_retries=5)
            if response_json:
                title = response_json.get("title", "unknown")
                description = response_json.get("description", "No description generated.")
                syslog2(LOG_NOTICE, "l2 topic named successfully", 
                       l2_id=topic.id, title=title, description_length=len(description))
                self.db.update_topic_l2_info(topic.id, title, description, vector_store=self.vector_store)
            else:
                syslog2(LOG_WARNING, "failed to name l2 topic after retries, setting to unknown", 
                       l2_id=topic.id, chunks_used=len(selected_chunks_texts))
                self.db.update_topic_l2_info(topic.id, "unknown", "Failed to generate description after retries.", vector_store=self.vector_store)
        except Exception as e:
            syslog2(LOG_ERR, "failed to name l2 topic", l2_id=topic.id, error=str(e), chunks_used=len(selected_chunks_texts))
            try:
                self.db.update_topic_l2_info(topic.id, "unknown", f"Error: {str(e)}", vector_store=self.vector_store)
            except Exception:
                pass

    def _call_llm_json(self, prompt: str, max_retries: int = 5) -> Optional[Dict]:
        """
        Helper to call LLM and parse JSON response with retry logic.
        """
        response = None
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                syslog2(LOG_DEBUG, "calling llm for json response", attempt=attempt + 1, max_retries=max_retries, prompt_length=len(prompt))
                response = self.llm_client.complete(messages)
                
                if not response:
                    syslog2(LOG_WARNING, "llm returned empty response", attempt=attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                syslog2(LOG_DEBUG, "llm response received", attempt=attempt + 1, response_length=len(response), response_preview=response[:200])

                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                    syslog2(LOG_DEBUG, "extracted json from markdown code block", extracted_length=len(clean_response))
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                    syslog2(LOG_DEBUG, "extracted json from code block", extracted_length=len(clean_response))

                parsed_json = json.loads(clean_response)
                syslog2(LOG_DEBUG, "successfully parsed json from llm", attempt=attempt + 1, keys=list(parsed_json.keys()) if isinstance(parsed_json, dict) else "not a dict")
                
                # Validate JSON structure and required fields
                if not isinstance(parsed_json, dict):
                    syslog2(LOG_WARNING, "llm response is not a dict, retrying", 
                           attempt=attempt + 1, max_retries=max_retries,
                           response_type=type(parsed_json).__name__)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                # Check for required keys
                required_keys = ['title', 'description']
                missing_keys = [key for key in required_keys if key not in parsed_json]
                if missing_keys:
                    syslog2(LOG_WARNING, "llm response missing required keys, retrying", 
                           attempt=attempt + 1, max_retries=max_retries,
                           missing_keys=missing_keys, available_keys=list(parsed_json.keys()))
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                # Check for empty values
                title = parsed_json.get('title', '').strip() if parsed_json.get('title') else ''
                description = parsed_json.get('description', '').strip() if parsed_json.get('description') else ''
                
                if not title or not description:
                    syslog2(LOG_WARNING, "llm response has empty required fields, retrying", 
                           attempt=attempt + 1, max_retries=max_retries,
                           title_empty=not title, description_empty=not description,
                           title_preview=title[:50] if title else "empty",
                           description_preview=description[:50] if description else "empty")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                return parsed_json

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    response_preview = response[:500] if response else "N/A"
                    syslog2(LOG_WARNING, "failed to parse json from llm, retrying",
                            attempt=attempt + 1, max_retries=max_retries, 
                            error=str(e), response_preview=response_preview,
                            response_length=len(response) if response else 0)
                    time.sleep(2 ** attempt)
                else:
                    response_preview = response[:1000] if response else "N/A"
                    syslog2(LOG_WARNING, "failed to parse json from llm after all retries",
                            error=str(e), response_preview=response_preview,
                            response_length=len(response) if response else 0,
                            full_response=response if response and len(response) < 2000 else None)
                    return None

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate limit" in error_str or "rate-limited" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    delay = 2 ** attempt
                    syslog2(LOG_WARNING, "rate limit error, retrying with exponential backoff",
                            attempt=attempt + 1, max_retries=max_retries, delay=delay, error=str(e)[:100])
                    time.sleep(delay)
                elif attempt < max_retries - 1:
                    syslog2(LOG_WARNING, "llm call error, retrying",
                            attempt=attempt + 1, max_retries=max_retries, error=str(e)[:100])
                    time.sleep(1)
                else:
                    syslog2(LOG_ERR, "llm call error after all retries", error=str(e))
                    return None

        return None

    def _cosine_similarity(self, v1, v2) -> float:
        """cosine similarity with zero-norm protection"""
        a = np.array(v1, dtype=float).ravel()
        b = np.array(v2, dtype=float).ravel()
        if a.size != b.size:
            n = min(a.size, b.size)
            a = a[:n]
            b = b[:n]
        norm1 = np.linalg.norm(a)
        norm2 = np.linalg.norm(b)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm1 * norm2))

    def _get_chunk_embeddings(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        batch load embeddings for given chunk ids
        expects vector_store.get_embeddings_by_ids to return dict with 'ids' and 'embeddings'
        """
        if not chunk_ids:
            return {}

        try:
            data = self.vector_store.get_embeddings_by_ids(chunk_ids)
        except AttributeError:
            syslog2(LOG_ERR, "vector_store missing get_embeddings_by_ids", count=len(chunk_ids))
            return {}
        except Exception as e:
            syslog2(LOG_ERR, "failed to load chunk embeddings", error=str(e)[:100])
            return {}

        ids = data.get("ids") or []
        embeddings = data.get("embeddings")
        
        # Safe check for empty embeddings (handles None, list, and numpy arrays)
        ids_empty = not ids or len(ids) == 0
        embeddings_empty = (embeddings is None or 
                           (isinstance(embeddings, list) and len(embeddings) == 0) or
                           (hasattr(embeddings, 'size') and embeddings.size == 0))
        
        if ids_empty or embeddings_empty:
            return {}

        arr = np.array(embeddings, dtype=float)
        mapping: Dict[str, np.ndarray] = {}
        for cid, vec in zip(ids, arr):
            mapping[cid] = vec

        return mapping
