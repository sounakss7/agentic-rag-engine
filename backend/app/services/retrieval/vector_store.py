"""
NexusRAG Vector Store Client.
Connects to Qdrant Cloud with batched embeddings or runs a resilient in-memory Cosine Similarity fallback.
"""

import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    IS_QDRANT_STUB = False
except ImportError:
    IS_QDRANT_STUB = True
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
        def __init__(self, *args, **kwargs):
            self.store: List[PointStruct] = []

        def get_collections(self):
            class Col: name = "nexus_knowledge_base"
            class Cols: collections = [Col()]
            return Cols()

        def recreate_collection(self, collection_name, vectors_config):
            self.store = []
            return True

        def upsert(self, collection_name, points):
            self.store.extend(points)
            return True

        def query_points(self, collection_name, query, limit=10):
            class Match:
                def __init__(self, payload, score):
                    self.payload = payload
                    self.score = score

            class Res:
                def __init__(self, points):
                    self.points = points
                def __iter__(self):
                    return iter(self.points)
                def __len__(self):
                    return len(self.points)
                def __getitem__(self, idx):
                    return self.points[idx]

            if not self.store or query is None:
                return Res([])

            q_vec = np.array(query, dtype=float)
            q_norm = np.linalg.norm(q_vec)
            scored = []

            for p in self.store:
                if hasattr(p, 'vector') and p.vector is not None:
                    p_vec = np.array(p.vector, dtype=float)
                    p_norm = np.linalg.norm(p_vec)
                    cos = float(np.dot(q_vec, p_vec) / (q_norm * p_norm)) if (q_norm > 0 and p_norm > 0) else 0.0
                else:
                    cos = 0.0
                scored.append(Match(p.payload, cos))

            scored.sort(key=lambda x: x.score, reverse=True)
            return Res(scored[:limit])

from backend.app.core.config import settings, register_degraded, clear_degraded
from backend.app.core.logging import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class VectorStoreService:
    """Manages dense vector indexing and similarity search in Qdrant Cloud or local memory."""

    def __init__(self):
        self._genai_client = None
        self.qdrant_client = self._init_qdrant()

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def _init_qdrant(self) -> QdrantClient:
        """Initializes connection to Qdrant Cloud or local in-memory store."""
        if settings.active_qdrant_url and settings.active_qdrant_key and not IS_QDRANT_STUB:
            try:
                client = QdrantClient(
                    url=settings.active_qdrant_url,
                    api_key=settings.active_qdrant_key,
                    timeout=15.0,
                    check_compatibility=False
                )
                client.get_collections()
                logger.info(f"Connected to Qdrant Cloud at {settings.active_qdrant_url}")
                clear_degraded("Qdrant Vector DB")
                return client
            except Exception as e:
                logger.warning(f"Qdrant Cloud connection failed: {e}. Falling back to in-memory store.")

        register_degraded("Qdrant Vector DB", "Running in-memory Cosine store")
        return QdrantClient(":memory:")

    def get_embedding(self, text: str) -> List[float]:
        """Generates 768-dimensional embedding vector via gemini-embedding-001 or SHA256 fallback."""
        client = self._get_genai_client()
        if client:
            try:
                emb_cfg = types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIM) if types else None
                res = client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=text,
                    config=emb_cfg
                )
                if hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                    return list(res.embedding.values)
                elif hasattr(res, 'embeddings') and res.embeddings:
                    return list(res.embeddings[0].values)
            except Exception as e:
                logger.warning(f"Gemini embedding API error: {e}")

        # Deterministic SHA-256 fallback
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        v = np.random.randn(settings.EMBEDDING_DIM)
        return (v / np.linalg.norm(v)).tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch generation of embeddings to minimize HTTP round-trips."""
        if not texts:
            return []

        client = self._get_genai_client()
        if client:
            batch_size = 32
            emb_cfg = types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIM) if types else None
            results: List[List[float]] = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    res = client.models.embed_content(
                        model=settings.EMBEDDING_MODEL,
                        contents=batch,
                        config=emb_cfg
                    )
                    batch_embs = []
                    if hasattr(res, 'embeddings') and res.embeddings:
                        for item in res.embeddings:
                            batch_embs.append(list(item.values))
                    elif hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                        batch_embs.append(list(res.embedding.values))

                    if len(batch_embs) == len(batch):
                        results.extend(batch_embs)
                        continue
                except Exception as e:
                    logger.warning(f"Batch embedding failed: {e}. Falling back to single calls.")

                for t in batch:
                    results.append(self.get_embedding(t))
            return results

        return [self.get_embedding(t) for t in texts]

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes child chunks into Qdrant collection."""
        if not chunks:
            return 0

        # Recreate collection cleanly
        try:
            self.qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE)
            )
        except Exception:
            pass

        texts = [c["content"] for c in chunks]
        embeddings = self.get_embeddings_batch(texts)

        points = []
        for idx, chunk in enumerate(chunks):
            vec = embeddings[idx] if idx < len(embeddings) else self.get_embedding(chunk["content"])
            points.append(PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "parent_id": chunk.get("parent_id", ""),
                    "parent_content": chunk.get("parent_content", ""),
                    "metadata": chunk.get("metadata", {})
                }
            ))

        batch_size = 64
        for i in range(0, len(points), batch_size):
            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points[i:i + batch_size]
            )

        logger.info(f"Indexed {len(points)} vector chunks into Qdrant.")
        return len(points)

    def search(self, query: str, top_k: int = settings.TOP_K_VECTOR) -> List[Dict[str, Any]]:
        """Dense similarity search."""
        q_vec = self.get_embedding(query)
        results = []

        try:
            res = self.qdrant_client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=q_vec,
                limit=top_k
            )
            pts = getattr(res, "points", res) or []

            for rank, pt in enumerate(pts):
                payload = getattr(pt, "payload", {}) or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", f"chunk_{rank}"),
                    "content": payload.get("content", ""),
                    "parent_id": payload.get("parent_id", ""),
                    "parent_content": payload.get("parent_content", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": float(getattr(pt, "score", 0.85)),
                    "rank": rank + 1,
                    "search_type": "dense_vector"
                })
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        return results
