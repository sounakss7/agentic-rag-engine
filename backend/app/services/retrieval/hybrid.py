"""
NexusRAG 3-Way Hybrid Retrieval & Multi-Source Fusion Engine.
Fuses Qdrant Dense Vectors + Lucene-Smoothed BM25 + Knowledge Graph (GraphRAG) via Reciprocal Rank Fusion.
Resolves Parent Context Blocks for high-fidelity generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
        def get_scores(self, query_tokens):
            scores = []
            q_set = set(query_tokens)
            for doc in self.corpus:
                scores.append(float(len(q_set.intersection(set(doc)))))
            return np.array(scores)

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.retrieval.vector_store import VectorStoreService
from backend.app.services.retrieval.graph_store import GraphStoreService
from backend.app.services.retrieval.reranker import CrossEncoderReranker


class HybridRetrievalService:
    """
    Coordinates multi-modal retrieval across:
    1. Dense Semantic Vectors (Qdrant Cloud / In-Memory)
    2. Sparse Lexical Matching (BM25Okapi with Lucene Smoothing)
    3. Relational Triples & Subgraphs (GraphRAG Knowledge Graph)
    4. Cross-Encoder Reranking (FlashRank)
    """

    def __init__(
        self,
        vector_service: Optional[VectorStoreService] = None,
        graph_service: Optional[GraphStoreService] = None,
        reranker_service: Optional[CrossEncoderReranker] = None
    ):
        self.vector_store = vector_service or VectorStoreService()
        self.graph_store = graph_service or GraphStoreService()
        self.reranker = reranker_service or CrossEncoderReranker()

        self.child_chunks: List[Dict[str, Any]] = []
        self.parent_sections: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None

    def build_index(
        self,
        child_chunks: List[Dict[str, Any]],
        parent_sections: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Indexes chunks into BM25, Qdrant Vector Store, and GraphRAG Knowledge Graph.
        """
        self.child_chunks = child_chunks
        self.parent_sections = parent_sections

        if not child_chunks:
            return {"status": "empty", "indexed_chunks": 0, "graph_triples": 0}

        # 1. Build Sparse BM25 Index with Lucene Smoothing Floor
        corpus_tokens = [c["content"].lower().split() for c in child_chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        if hasattr(self.bm25, "idf"):
            # Enforce non-negative IDF floor to prevent zero-scores on small corpora
            self.bm25.idf = {k: max(v, 0.25) for k, v in self.bm25.idf.items()}

        # 2. Build Dense Qdrant Vector Store
        indexed_vectors = self.vector_store.index_chunks(child_chunks)

        # 3. Build GraphRAG Knowledge Graph from Parent Sections
        graph_triples = self.graph_store.build_graph_from_sections(parent_sections)

        logger.info(f"Hybrid Index complete: {indexed_vectors} vector chunks, {graph_triples} graph relations.")
        return {
            "status": "success",
            "indexed_chunks": len(child_chunks),
            "parent_sections": len(parent_sections),
            "graph_triples": graph_triples
        }

    def sparse_search(self, query: str, top_k: int = settings.TOP_K_SPARSE) -> List[Dict[str, Any]]:
        """BM25 lexical search using clean query terms."""
        if not self.bm25 or not self.child_chunks:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk = self.child_chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                chunk["rank"] = rank + 1
                chunk["search_type"] = "sparse_bm25"
                results.append(chunk)

        return results

    def hybrid_search(
        self,
        query: str,
        dense_query: Optional[str] = None,
        top_n: int = settings.TOP_N_RERANK
    ) -> List[Dict[str, Any]]:
        """
        Executes 3-way retrieval (Dense, Sparse, Graph), fuses results via RRF,
        reranks using FlashRank, and attaches Parent Section context.
        """
        if not self.child_chunks:
            return []

        clean_dense_query = dense_query or query

        # 1. Parallel branch retrieval
        dense_hits = self.vector_store.search(clean_dense_query, top_k=settings.TOP_K_VECTOR)
        sparse_hits = self.sparse_search(query, top_k=settings.TOP_K_SPARSE)
        graph_hits = self.graph_store.search_subgraph(query, top_k=settings.TOP_K_GRAPH)

        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        k = settings.RRF_K

        # Fuse Dense
        for rank, item in enumerate(dense_hits):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[cid] = item

        # Fuse Sparse
        for rank, item in enumerate(sparse_hits):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # Fuse Knowledge Graph Hits
        for rank, item in enumerate(graph_hits):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[cid] = item

        # Sort by composite RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        candidate_chunks = []
        for cid in sorted_cids[:15]:
            c = chunk_map[cid].copy()
            c["rrf_score"] = rrf_scores[cid]
            candidate_chunks.append(c)

        # 3. Cross-Encoder Reranking
        reranked = self.reranker.rerank(query, candidate_chunks, top_n=top_n)

        # 4. Resolve Parent Section Context
        for item in reranked:
            pid = item.get("parent_id")
            if pid and pid in self.parent_sections:
                parent = self.parent_sections[pid]
                item["parent_content"] = parent.get("content", "")
                item["is_table"] = parent.get("metadata", {}).get("is_table", False)

        return reranked
