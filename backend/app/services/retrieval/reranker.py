"""
NexusRAG FlashRank Cross-Encoder Reranker.
Re-scores and orders fused multi-source candidate passages with sub-millisecond latency.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None

from backend.app.core.config import settings, register_degraded, clear_degraded
from backend.app.core.logging import logger


class CrossEncoderReranker:
    """FlashRank cross-encoder reranker with resilient fallbacks."""

    def __init__(self):
        self.ranker = self._init_ranker()

    def _init_ranker(self) -> Optional[Any]:
        if Ranker is None:
            register_degraded("FlashRank Reranker", "flashrank library not found")
            return None

        cache_dir = os.path.join(tempfile.gettempdir(), "nexus_flashrank_cache")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            ranker = Ranker(model_name=settings.FLASHRANK_MODEL, cache_dir=cache_dir)
            clear_degraded("FlashRank Reranker")
            logger.info(f"Loaded FlashRank Cross-Encoder: {settings.FLASHRANK_MODEL}")
            return ranker
        except Exception as e:
            try:
                logger.info(f"Retrying FlashRank default model: {e}")
                ranker = Ranker(cache_dir=cache_dir)
                clear_degraded("FlashRank Reranker")
                return ranker
            except Exception as e2:
                logger.warning(f"FlashRank init failed: {e2}. Using score pass-through.")
                register_degraded("FlashRank Reranker", str(e2))
                return None

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_n: int = settings.TOP_N_RERANK) -> List[Dict[str, Any]]:
        """Reranks candidate chunks using the cross-encoder."""
        if not candidate_chunks:
            return []

        if len(candidate_chunks) <= top_n or self.ranker is None or RerankRequest is None:
            # Sort by existing score
            candidate_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidate_chunks[:top_n]

        try:
            passages = [
                {"id": idx, "text": chunk["content"], "meta": chunk}
                for idx, chunk in enumerate(candidate_chunks)
            ]
            req = RerankRequest(query=query, passages=passages)
            ranked_output = self.ranker.rerank(req)

            reranked_results = []
            for item in ranked_output[:top_n]:
                chunk = item["meta"].copy()
                chunk["score"] = float(item["score"])
                chunk["rerank_score"] = float(item["score"])
                reranked_results.append(chunk)

            return reranked_results
        except Exception as e:
            logger.warning(f"FlashRank reranking error: {e}. Falling back to initial ordering.")
            candidate_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidate_chunks[:top_n]
