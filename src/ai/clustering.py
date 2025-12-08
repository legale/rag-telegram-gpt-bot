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
MAX_L2_CHUNK_TEXT_LEN = 500


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
        syslog2(LOG_INFO, "starting l1 clustering",
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                method=cluster_selection_method)

        # 1. Fetch data
        data = self.vector_store.get_all_embeddings()
        ids = data['ids']
        embeddings = data['embeddings']
        metadatas = data['metadatas']

        if not ids or embeddings is None or (isinstance(embeddings, list) and not embeddings) or (hasattr(embeddings, 'size') and embeddings.size == 0):
            syslog2(LOG_WARNING, "no data for clustering")
            return

        X = np.array(embeddings)

        # Check dimensionality
        if len(X.shape) != 2:
            syslog2(LOG_ERR, "invalid embeddings shape", shape=str(X.shape))
            return

        n_samples = len(ids)
        syslog2(LOG_INFO, "clustering chunks", count=n_samples)

        # Validate min_cluster_size (HDBSCAN requires >= 2)
        if min_cluster_size < 2:
            syslog2(LOG_ERR, "min_cluster_size must be at least 2 (HDBSCAN requirement)",
                    provided=min_cluster_size)
            raise ValueError(f"min_cluster_size must be at least 2, got {min_cluster_size}")

        # Auto-adjust min_cluster_size if too large for dataset
        if min_cluster_size > n_samples:
            syslog2(LOG_WARNING, "min_cluster_size too large, adjusting",
                    original=min_cluster_size, adjusted=n_samples)
            min_cluster_size = max(2, n_samples // 2)

        # Normalize embeddings if using cosine metric
        # HDBSCAN doesn't support cosine directly, but euclidean on normalized vectors is equivalent
        actual_metric = metric
        if metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
            actual_metric = 'euclidean'
            syslog2(LOG_DEBUG, "normalized embeddings for cosine similarity, using euclidean metric")

        # 2. Run HDBSCAN
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

        # 3. Process clusters
        clusters: Dict[int, List[int]] = {}

        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        noise_count = len(clusters.get(-1, []))
        valid_clusters = len(clusters) - (1 if -1 in clusters else 0)
        noise_percentage = (noise_count / n_samples * 100) if n_samples > 0 else 0

        syslog2(LOG_INFO, "clustering complete",
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")

        if noise_percentage > 50:
            syslog2(LOG_WARNING, "high noise percentage, consider lowering min_cluster_size or min_samples",
                    noise_percentage=f"{noise_percentage:.1f}%")

        # 4. Save topics
        self.db.clear_topics_l1()
        self._l1_topic_assignments = {}

        for label, indices in clusters.items():
            if label == -1:
                # noise
                for idx in indices:
                    chunk_id = ids[idx]
                    if None not in self._l1_topic_assignments:
                        self._l1_topic_assignments[None] = []
                    self._l1_topic_assignments[None].append(chunk_id)
                continue

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
                ts_to=ts_to
            )

            self._l1_topic_assignments[topic_id] = cluster_ids

        syslog2(LOG_INFO, "l1 topics created", count=valid_clusters)

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

        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_chunks, desc="Assigning topics to chunks", unit="chunk")
            except ImportError:
                pbar = None
        else:
            pbar = None

        try:
            for topic_id, chunk_ids in self._l1_topic_assignments.items():
                if topic_id is None:
                    for chunk_id in chunk_ids:
                        self.db.update_chunk_topics(chunk_id, topic_l1_id=None, topic_l2_id=None)
                        # Update metadata in chroma_db
                        try:
                            self.vector_store.update_chunk_metadata(
                                chunk_id,
                                {"topic_l1_id": None, "topic_l2_id": None}
                            )
                        except Exception as e:
                            syslog2(LOG_DEBUG, "failed to update chunk metadata in chroma", chunk_id=chunk_id, error=str(e))
                        noise_count += 1
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({"assigned": assigned_count, "noise": noise_count})
                else:
                    for chunk_id in chunk_ids:
                        self.db.update_chunk_topics(chunk_id, topic_l1_id=topic_id, topic_l2_id=None)
                        # Update metadata in chroma_db
                        try:
                            # Get existing metadata first
                            existing = self.vector_store.get_embeddings_by_ids([chunk_id])
                            existing_meta = existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
                            existing_meta["topic_l1_id"] = topic_id
                            existing_meta["topic_l2_id"] = None  # Clear L2 if was set
                            self.vector_store.update_chunk_metadata(chunk_id, existing_meta)
                        except Exception as e:
                            syslog2(LOG_DEBUG, "failed to update chunk metadata in chroma", chunk_id=chunk_id, error=str(e))
                        assigned_count += 1
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({"assigned": assigned_count, "noise": noise_count})
        finally:
            if pbar:
                pbar.close()

        syslog2(LOG_INFO, "l1 topics assigned to chunks", assigned=assigned_count, noise=noise_count)
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
        syslog2(LOG_INFO, "starting l2 clustering")

        # 1. Fetch L1 topics
        l1_topics = self.db.get_all_topics_l1()
        if not l1_topics:
            syslog2(LOG_WARNING, "no l1 topics found for l2 clustering")
            return

        ids = []
        embeddings = []
        for t in l1_topics:
            if t.center_vec:
                try:
                    vec = json.loads(t.center_vec)
                    embeddings.append(vec)
                    ids.append(t.id)
                except Exception:
                    pass

        if not embeddings:
            syslog2(LOG_WARNING, "no l1 topic centroids found")
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

        syslog2(LOG_INFO, "clustering l1 topics", count=n_l1_topics)

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

        syslog2(LOG_INFO, "l2 clustering complete",
                clusters_found=valid_clusters,
                noise_count=noise_count,
                noise_percentage=f"{noise_percentage:.1f}%")

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(clusters), desc="Creating L2 topics", unit="cluster")
        except ImportError:
            pbar = None

        try:
            for label, indices in clusters.items():
                if label == -1:
                    # noise
                    for idx in indices:
                        l1_id = ids[idx]
                        self.db.update_topic_l1_parent(l1_id, parent_l2_id=None)
                        self.db.update_chunks_parent_l2(l1_id, topic_l2_id=None)
                    if pbar:
                        pbar.update(1)
                    continue

                cluster_indices = indices
                cluster_l1_ids = [ids[i] for i in cluster_indices]
                cluster_embeddings = X[cluster_indices]

                if pbar:
                    pbar.set_postfix({"cluster": label, "l1_topics": len(cluster_l1_ids)})

                centroid = np.mean(cluster_embeddings, axis=0).tolist()

                total_chunk_count = 0
                for i in cluster_indices:
                    l1_topic = next(t for t in l1_topics if t.id == ids[i])
                    total_chunk_count += l1_topic.chunk_count

                l2_id = self.db.create_topic_l2(
                    title="unknown",
                    descr="Pending description...",
                    chunk_count=total_chunk_count,
                    center_vec=centroid,
                    vector_store=self.vector_store
                )

                for l1_id in cluster_l1_ids:
                    self.db.update_topic_l1_parent(l1_id, parent_l2_id=l2_id)
                    # Update chunks in SQLite
                    updated_count = self.db.update_chunks_parent_l2(l1_id, topic_l2_id=l2_id)
                    # Update metadata in chroma_db for all affected chunks
                    if updated_count > 0:
                        try:
                            chunks = self.db.get_chunks_by_topic_l1(l1_id)
                            for chunk in chunks:
                                try:
                                    # Get existing metadata
                                    existing = self.vector_store.get_embeddings_by_ids([chunk.id])
                                    existing_meta = existing.get("metadatas", [{}])[0] if existing.get("metadatas") else {}
                                    existing_meta["topic_l2_id"] = l2_id
                                    if chunk.topic_l1_id:
                                        existing_meta["topic_l1_id"] = chunk.topic_l1_id
                                    self.vector_store.update_chunk_metadata(chunk.id, existing_meta)
                                except Exception as e:
                                    syslog2(LOG_DEBUG, "failed to update chunk metadata in chroma", chunk_id=chunk.id, error=str(e))
                        except Exception as e:
                            syslog2(LOG_WARNING, "failed to update chunk metadata in chroma for l2 assignment", l1_id=l1_id, l2_id=l2_id, error=str(e))

                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        syslog2(LOG_INFO, "l2 topics saved")

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
            syslog2(LOG_INFO, "naming l1 topics", filtered=total_l1_filtered, total=total_l1_all)

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
            syslog2(LOG_INFO, "naming l2 topics", filtered=total_l2_filtered, total=total_l2_all)

            for idx, topic in enumerate(l2_topics, 1):
                self._name_l2_topic(topic)
                if progress_callback:
                    progress_callback(idx, total_l2_filtered, 'l2', total_all=total_l2_all)

        syslog2(LOG_INFO, "topic naming complete")

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
        # l2 center
        if not topic.center_vec:
            return

        try:
            l2_center_vec = np.array(json.loads(topic.center_vec), dtype=float)
        except Exception:
            syslog2(LOG_WARNING, "invalid l2 center_vec, skipping naming", id=topic.id)
            return

        subtopics = self.db.get_l1_topics_by_l2(topic.id)
        if not subtopics:
            return

        # precompute l1 centers
        l1_centers: Dict[int, np.ndarray] = {}
        l1_scored: List[Tuple[float, object]] = []

        for t in subtopics:
            if not t.center_vec:
                continue
            try:
                vec = np.array(json.loads(t.center_vec), dtype=float)
            except Exception:
                continue
            l1_centers[t.id] = vec
            sim = self._cosine_similarity(l2_center_vec, vec)
            l1_scored.append((sim, t))

        if not l1_scored:
            syslog2(LOG_WARNING, "no valid l1 centers for l2 naming", id=topic.id)
            return

        l1_scored.sort(key=lambda x: x[0], reverse=True)
        top_l1_topics = [t for _, t in l1_scored[:MAX_L1_TOPICS_FOR_L2_NAMING]]

        selected_chunks_texts: List[str] = []
        total_limit = MAX_TOTAL_CHUNKS_FOR_L2_NAMING

        for l1_topic in top_l1_topics:
            if len(selected_chunks_texts) >= total_limit:
                break

            chunks = self.db.get_chunks_by_topic_l1(l1_topic.id)
            if not chunks:
                continue

            chunk_ids = [c.id for c in chunks if getattr(c, "id", None) is not None]
            if not chunk_ids:
                continue

            emb_map = self._get_chunk_embeddings(chunk_ids)
            if not emb_map:
                continue

            l1_center = l1_centers.get(l1_topic.id)
            if l1_center is None:
                continue

            scored_chunks: List[Tuple[float, object]] = []
            for c in chunks:
                cid = getattr(c, "id", None)
                if cid is None:
                    continue
                vec = emb_map.get(cid)
                if vec is None:
                    continue
                sim = self._cosine_similarity(l1_center, vec)
                scored_chunks.append((sim, c))

            if not scored_chunks:
                continue

            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            per_l1_limit = min(MAX_REPRESENTATIVE_CHUNKS_PER_L1,
                               total_limit - len(selected_chunks_texts))

            for sim, c in scored_chunks[:per_l1_limit]:
                text = c.text if hasattr(c, "text") and c.text else ""
                if not text:
                    continue
                trimmed = text[:MAX_L2_CHUNK_TEXT_LEN]
                block = f"--- chunk from topic l1-{l1_topic.id} ---\n{trimmed}"
                selected_chunks_texts.append(block)

                if len(selected_chunks_texts) >= total_limit:
                    break

        if not selected_chunks_texts:
            syslog2(LOG_WARNING, "no representative chunks selected for l2 naming", id=topic.id)
            return

        chunks_text = "\n\n".join(selected_chunks_texts)
        prompt = TOPIC_L2_NAMING_PROMPT.format(subtopics=chunks_text)

        try:
            response_json = self._call_llm_json(prompt, max_retries=5)
            if response_json:
                title = response_json.get("title", "unknown")
                description = response_json.get("description", "No description generated.")
                self.db.update_topic_l2_info(topic.id, title, description, vector_store=self.vector_store)
            else:
                syslog2(LOG_WARNING, "failed to name l2 topic after retries, setting to unknown", id=topic.id)
                self.db.update_topic_l2_info(topic.id, "unknown", "Failed to generate description after retries.", vector_store=self.vector_store)
        except Exception as e:
            syslog2(LOG_ERR, "failed to name l2 topic", id=topic.id, error=str(e))
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
                response = self.llm_client.complete(messages)

                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()

                return json.loads(clean_response)

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    response_preview = response[:100] if response else "N/A"
                    syslog2(LOG_WARNING, "failed to parse json from llm, retrying",
                            attempt=attempt + 1, max_retries=max_retries, response=response_preview)
                    time.sleep(2 ** attempt)
                else:
                    response_preview = response[:100] if response else "N/A"
                    syslog2(LOG_WARNING, "failed to parse json from llm after all retries",
                            response=response_preview)
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
        embeddings = data.get("embeddings") or []

        if not ids or not embeddings:
            return {}

        arr = np.array(embeddings, dtype=float)
        mapping: Dict[str, np.ndarray] = {}
        for cid, vec in zip(ids, arr):
            mapping[cid] = vec

        return mapping
