import os
import tempfile
import math
import logging
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

from config import config, register_degraded_component, clear_degraded_component

logger = logging.getLogger("HybridRetriever")
logger.setLevel(logging.INFO)

# Qdrant Vector Store with Cosine Similarity Fallback Stub
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
    IS_QDRANT_STUB = False
except ImportError as e:
    IS_QDRANT_STUB = True
    print(f"[DEGRADED] QdrantClient: {type(e).__name__}: {e}")
    register_degraded_component("QdrantClient (stub)")
    
    class VectorParams:
        def __init__(self, size=768, distance=None):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload

    class Distance:
        COSINE = "COSINE"

    class QdrantClient:
        """
        Fallback Qdrant In-Memory Vector Store performing exact Cosine Similarity search.
        Used when the qdrant-client package is not installed.
        """
        def __init__(self, *args, **kwargs):
            self.store: List[PointStruct] = []
            logger.warning("[DEGRADED STUB] qdrant-client package missing. Running in-memory Cosine Similarity vector store.")

        def get_collections(self):
            class Col: name = config.QDRANT_COLLECTION
            class Cols: collections = [Col()]
            return Cols()

        def create_collection(self, **kwargs): pass
        
        def recreate_collection(self, **kwargs):
            self.store = []

        def upsert(self, collection_name, points):
            self.store.extend(points)

        def search(self, collection_name, query_vector, limit=10):
            return self.query_points(collection_name, query_vector, limit)

        def query_points(self, collection_name, query, limit=10):
            class Match:
                def __init__(self, payload, score):
                    self.payload = payload
                    self.score = score

            if not self.store or not query:
                class Res: points = []
                return Res()

            q_vec = np.array(query, dtype=float)
            q_norm = np.linalg.norm(q_vec)
            scored_points = []

            for p in self.store:
                if hasattr(p, 'vector') and p.vector is not None:
                    p_vec = np.array(p.vector, dtype=float)
                    p_norm = np.linalg.norm(p_vec)
                    if q_norm > 0 and p_norm > 0:
                        cos_sim = float(np.dot(q_vec, p_vec) / (q_norm * p_norm))
                    else:
                        cos_sim = 0.0
                else:
                    cos_sim = 0.0
                scored_points.append(Match(p.payload, cos_sim))

            scored_points.sort(key=lambda x: x.score, reverse=True)

            class Res:
                points = scored_points[:limit]
            return Res()


# Rank-BM25 Sparse Retriever with fallback
try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"[DEGRADED] BM25Okapi: {type(e).__name__}: {e}")
    register_degraded_component("BM25Okapi (stub)")
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query_tokens):
            scores = []
            q_set = set(query_tokens)
            for doc in self.corpus:
                d_set = set(doc)
                match = len(q_set.intersection(d_set))
                scores.append(float(match))
            return np.array(scores)


# FlashRank Cross-Encoder Reranker
try:
    from flashrank import Ranker, RerankRequest
except ImportError as e:
    print(f"[DEGRADED] FlashRank Import: {type(e).__name__}: {e}")
    register_degraded_component("FlashRank (stub)")
    Ranker = None
    RerankRequest = None


# Google GenAI SDK
try:
    from google import genai
except ImportError as e:
    print(f"[DEGRADED] Google GenAI Import: {type(e).__name__}: {e}")
    register_degraded_component("Gemini GenAI (stub)")
    genai = None


class HybridRetriever:
    """
    Hybrid Retrieval Engine combining:
    1. BM25 Sparse Keyword Search
    2. Qdrant Dense Vector Search (Cloud or In-Memory Cosine Similarity)
    3. Reciprocal Rank Fusion (RRF)
    4. FlashRank Cross-Encoder Reranking
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.flashrank_reranker: Optional[Any] = None

        self._init_qdrant()
        self._init_flashrank()

    def _get_genai_client(self) -> Optional[Any]:
        """Dynamically retrieves GenAI client using active GEMINI_API_KEY."""
        if genai is not None and config.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as e:
                err_line = f"[DEGRADED] Gemini GenAI Client Init: {type(e).__name__}: {e}"
                print(err_line)
                logger.warning(err_line)
        return None

    def _init_qdrant(self) -> None:
        """Initializes Qdrant client (Cloud or In-Memory fallback)."""
        try:
            if config.QDRANT_URL and config.QDRANT_API_KEY and not IS_QDRANT_STUB:
                self.qdrant_client = QdrantClient(
                    url=config.QDRANT_URL,
                    api_key=config.QDRANT_API_KEY
                )
                logger.info(f"Connected to Qdrant Cloud at {config.QDRANT_URL}")
            else:
                self.qdrant_client = QdrantClient(":memory:")
                if not IS_QDRANT_STUB and not config.QDRANT_URL:
                    register_degraded_component("Qdrant Local (:memory:)")
                logger.info("Initialized Qdrant in Local In-Memory mode.")

            self._ensure_collection()
        except Exception as e:
            err_line = f"[DEGRADED] Qdrant Cloud Connection: {type(e).__name__}: {e}"
            print(err_line)
            logger.warning(err_line)
            self.qdrant_client = QdrantClient(":memory:")
            register_degraded_component("Qdrant Local (:memory:)")
            self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensures Qdrant collection exists."""
        if not self.qdrant_client:
            return
        
        try:
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if config.QDRANT_COLLECTION not in collections:
                self.qdrant_client.create_collection(
                    collection_name=config.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.warning(f"Qdrant collection creation notice: {e}")

    def _init_flashrank(self) -> None:
        """Initializes FlashRank cross-encoder reranker with writable temp cache dir fallback."""
        if Ranker is not None:
            try:
                cache_dir = os.path.join(tempfile.gettempdir(), "flashrank_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                try:
                    self.flashrank_reranker = Ranker(model_name=config.FLASHRANK_MODEL, cache_dir=cache_dir)
                except TypeError:
                    self.flashrank_reranker = Ranker(model_name=config.FLASHRANK_MODEL)

                logger.info(f"Loaded FlashRank Reranker: {config.FLASHRANK_MODEL}")
                clear_degraded_component("FlashRank (stub)")
            except Exception as e:
                err_line = f"[DEGRADED] FlashRank Reranker Init: {type(e).__name__}: {e}"
                print(err_line)
                logger.warning(err_line)
                register_degraded_component("FlashRank (stub)")
                self.flashrank_reranker = None
        else:
            err_line = "[DEGRADED] FlashRank Reranker: ImportError: flashrank package not installed"
            print(err_line)
            register_degraded_component("FlashRank (stub)")
            self.flashrank_reranker = None

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates 768-dimensional embedding vector for input text via Gemini API text-embedding-004.
        If Gemini API key is missing or call fails, falls back to deterministic SHA-256 vector generation.
        """
        client = self._get_genai_client()
        if client:
            try:
                response = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=text
                )
                if hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
                    clear_degraded_component("Gemini Embeddings (SHA256 fallback)")
                    return list(response.embedding.values)
                elif hasattr(response, 'embeddings') and response.embeddings:
                    clear_degraded_component("Gemini Embeddings (SHA256 fallback)")
                    return list(response.embeddings[0].values)
            except Exception as e:
                err_line = f"[DEGRADED] Gemini Embeddings Call: {type(e).__name__}: {e}"
                print(err_line)
                logger.warning(err_line)
        else:
            err_line = "[DEGRADED] Gemini Embeddings Call: AuthError: GEMINI_API_KEY is missing or unresolved"
            print(err_line)
            logger.warning(err_line)

        # Stable, Process-Invariant SHA-256 Fallback Vector Generation
        register_degraded_component("Gemini Embeddings (SHA256 fallback)")
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        vec = np.random.randn(config.EMBEDDING_DIM)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist()


    def build_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexes chunks into both BM25 and Qdrant Vector Store.
        """
        self.chunks = chunks
        if not chunks:
            return {"indexed_count": 0, "status": "empty"}

        # 1. Build Sparse BM25 Index
        corpus_tokens = [chunk["content"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        # 2. Build Qdrant Dense Index
        self._ensure_collection()
        
        try:
            self.qdrant_client.recreate_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass

        points = []
        for idx, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk["content"])
            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
            ))

        # Upload points
        batch_size = 64
        for i in range(0, len(points), batch_size):
            self.qdrant_client.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=points[i:i + batch_size]
            )

        logger.info(f"Successfully indexed {len(chunks)} chunks into BM25 and Qdrant.")
        return {
            "indexed_count": len(chunks),
            "status": "success",
            "qdrant_collection": config.QDRANT_COLLECTION
        }

    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 keyword search."""
        if not self.bm25 or not self.chunks:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                    "search_type": "sparse"
                })
        return results

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Qdrant Vector Similarity search."""
        if not self.qdrant_client or not self.chunks:
            return []

        query_vector = self.get_embedding(query)
        results = []

        retrieval_mode = "stub" if IS_QDRANT_STUB else ("cloud" if config.QDRANT_URL else "memory")

        try:
            if hasattr(self.qdrant_client, 'search'):
                search_result = self.qdrant_client.search(
                    collection_name=config.QDRANT_COLLECTION,
                    query_vector=query_vector,
                    limit=top_k
                )
            elif hasattr(self.qdrant_client, 'query_points'):
                res = self.qdrant_client.query_points(
                    collection_name=config.QDRANT_COLLECTION,
                    query=query_vector,
                    limit=top_k
                )
                search_result = getattr(res, 'points', [])
            else:
                search_result = []

            for rank, point in enumerate(search_result):
                payload = getattr(point, 'payload', {}) or {}
                score = getattr(point, 'score', 0.85)
                results.append({
                    "chunk_id": payload.get("chunk_id", f"chunk_{rank}"),
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": float(score),
                    "rank": rank + 1,
                    "search_type": "dense",
                    "retrieval_mode": retrieval_mode
                })
        except Exception as e:
            logger.warning(f"Dense vector search warning: {e}")

        return results

    def reciprocal_rank_fusion(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        k: int = 60,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combines sparse and dense ranked lists using Reciprocal Rank Fusion (RRF).
        RRF_Score = sum(1 / (k + rank))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process Sparse
        for item in sparse_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            chunk_map[cid] = item

        # Process Dense
        for item in dense_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results = []
        for cid in sorted_cids[:top_n]:
            item = chunk_map[cid].copy()
            item["rrf_score"] = float(rrf_scores[cid])
            fused_results.append(item)

        return fused_results

    def flashrank_rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks candidates using FlashRank Cross-Encoder.
        """
        if not candidates:
            return []

        if self.flashrank_reranker is not None and RerankRequest is not None:
            try:
                passages = [
                    {"id": idx, "text": item["content"], "meta": item["metadata"]}
                    for idx, item in enumerate(candidates)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                ranked_result = self.flashrank_reranker.rerank(rerank_req)

                reranked = []
                for item in ranked_result[:top_n]:
                    orig_idx = item["id"]
                    cand = candidates[orig_idx].copy()
                    cand["rerank_score"] = float(item["score"])
                    reranked.append(cand)
                return reranked
            except Exception as e:
                logger.warning(f"FlashRank reranking error: {e}. Returning RRF top_n.")
                register_degraded_component("FlashRank (stub)")

        # Fallback if FlashRank is unavailable
        register_degraded_component("FlashRank (stub)")
        for item in candidates:
            item["rerank_score"] = item.get("rrf_score", 0.5)
        return candidates[:top_n]

    def search(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Full Hybrid Retrieval Pipeline:
        Sparse (BM25) + Dense (Qdrant) -> RRF Fusion -> FlashRank Rerank
        """
        sparse_res = self.sparse_search(query, top_k=config.TOP_K_SPARSE)
        dense_res = self.dense_search(query, top_k=config.TOP_K_DENSE)
        fused_res = self.reciprocal_rank_fusion(sparse_res, dense_res, k=config.RRF_K, top_n=10)
        final_reranked = self.flashrank_rerank(query, fused_res, top_n=top_n)
        return final_reranked

    def clear(self) -> None:
        """Clears index in memory and Qdrant."""
        self.chunks = []
        self.bm25 = None
        if self.qdrant_client:
            try:
                self.qdrant_client.recreate_collection(
                    collection_name=config.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=config.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
            except Exception as e:
                logger.warning(f"Error clearing Qdrant collection: {e}")
