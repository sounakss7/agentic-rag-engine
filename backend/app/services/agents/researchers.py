"""
NexusRAG Specialist Research & Context Grading Agents.
Executes parallel multi-source retrieval (Vector + Graph) and batched relevance grading.
"""

import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.retrieval.hybrid import HybridRetrievalService

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="1-based index of candidate chunk")
    is_relevant: bool = Field(description="True if chunk contains factual signal for answering user query")
    relevance_score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Short rationale for grade")


class GradeEvaluationReport(BaseModel):
    grades: List[ChunkGrade] = Field(default_factory=list)


class ResearchSpecialists:
    """Coordinates specialist research workers for Vector/Graph search and Web Fallback."""

    def __init__(self, hybrid_service: HybridRetrievalService):
        self.hybrid_service = hybrid_service
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def execute_vector_graph_search(
        self, sub_tasks: List[str], hyde_expansion: str
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval across all decomposed sub-queries and aggregates unique chunks.
        """
        all_chunks: List[Dict[str, Any]] = []
        seen_ids = set()

        for sub_q in sub_tasks:
            dense_q = f"{sub_q} {hyde_expansion[:200]}"
            hits = self.hybrid_service.hybrid_search(
                query=sub_q,
                dense_query=dense_q,
                top_n=settings.TOP_N_RERANK
            )
            for h in hits:
                cid = h.get("chunk_id")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(h)

        return all_chunks

    def grade_context_batch(
        self, query: str, candidate_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Grades all candidate chunks in 1 single batched LLM judge call.
        Returns:
            Tuple of (graded_and_filtered_chunks, fallback_required: bool)
        """
        if not candidate_chunks:
            return [], True

        client = self._get_genai_client()
        graded_results: List[Dict[str, Any]] = []

        if client and types is not None:
            formatted_chunks = "\n\n".join([
                f"--- [Chunk {idx+1} | Source: {c.get('metadata', {}).get('source', 'doc')}] ---\n{c['content']}"
                for idx, c in enumerate(candidate_chunks)
            ])

            prompt = (
                "You are an impartial, strict relevance evaluator.\n"
                f"User Question: {query}\n\n"
                f"Candidate Context Chunks:\n{formatted_chunks}\n\n"
                f"For each of the {len(candidate_chunks)} chunks, evaluate chunk_index, is_relevant, relevance_score (0.0 to 1.0), and reasoning."
            )

            for model_name in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GradeEvaluationReport
                        )
                    )
                    if res and res.text:
                        parsed = json.loads(res.text.strip().replace("```json", "").replace("```", ""))
                        grades = parsed.get("grades", [])
                        grade_map = {g.get("chunk_index", i+1): g for i, g in enumerate(grades)}

                        for idx, chunk in enumerate(candidate_chunks):
                            info = grade_map.get(idx + 1, {})
                            score = float(info.get("relevance_score", 0.75))
                            is_rel = bool(info.get("is_relevant", True))
                            reason = info.get("reasoning", "Batched evaluation")

                            c = chunk.copy()
                            c["relevance_score"] = score
                            c["is_relevant"] = is_rel
                            c["grading_reason"] = reason

                            if is_rel and score >= settings.RELEVANCE_THRESHOLD:
                                graded_results.append(c)
                        break
                except Exception as e:
                    if "429" in str(e):
                        continue
                    logger.warning(f"Grading LLM notice: {e}")
                    break

        # Fallback to lexical token overlap if LLM failed
        if not graded_results:
            stop_words = {"what", "is", "the", "a", "an", "in", "to", "for", "and", "of", "on", "by", "with"}
            q_terms = set(w for w in query.lower().split() if w not in stop_words) or set(query.lower().split())

            for c in candidate_chunks:
                doc_terms = set(c["content"].lower().split())
                overlap = len(q_terms.intersection(doc_terms))
                score = min(1.0, round(overlap / max(1, len(q_terms)), 2))
                is_rel = overlap > 0 and score >= 0.25

                chunk = c.copy()
                chunk["relevance_score"] = score
                chunk["is_relevant"] = is_rel
                chunk["grading_reason"] = f"Lexical overlap ({overlap} matches)"
                if is_rel:
                    graded_results.append(chunk)

        fallback_needed = len(graded_results) == 0
        return graded_results, fallback_needed

    def execute_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes query rewriting and live web research using Tavily API.
        """
        web_results = []
        if not settings.active_tavily_key or TavilyClient is None:
            return []

        # 1. Rewrite query for web retrieval
        client = self._get_genai_client()
        search_query = query
        if client:
            prompt = f"Rewrite this question into a clean search query for web retrieval:\nQuestion: {query}\nSearch Query:"
            try:
                for model in settings.CASCADE_MODELS:
                    try:
                        res = client.models.generate_content(model=model, contents=prompt)
                        if res and res.text:
                            search_query = res.text.strip().replace('"', "")
                            break
                    except Exception as e:
                        if "429" in str(e):
                            continue
                        break
            except Exception:
                pass

        # 2. Query Tavily
        try:
            tavily = TavilyClient(api_key=settings.active_tavily_key)
            resp = tavily.search(query=search_query, search_depth="basic", max_results=4)
            for idx, r in enumerate(resp.get("results", [])):
                web_results.append({
                    "chunk_id": f"tavily_web_{idx}",
                    "content": f"Title: {r.get('title', '')}\nSnippet: {r.get('content', '')}",
                    "parent_content": f"Title: {r.get('title', '')}\nSnippet: {r.get('content', '')}\nURL: {r.get('url', '')}",
                    "metadata": {
                        "source": f"Web Grounding: {r.get('url', 'web')}",
                        "page": 1,
                        "url": r.get("url", ""),
                        "is_web": True
                    },
                    "relevance_score": 0.90,
                    "is_relevant": True
                })
        except Exception as e:
            logger.warning(f"Tavily search error: {e}")

        return web_results
