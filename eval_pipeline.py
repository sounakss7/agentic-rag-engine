import time
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
try:
    from google import genai
except ImportError:
    genai = None

from config import config, is_degraded_mode, get_degraded_components

logger = logging.getLogger("RAGASEvaluator")
logger.setLevel(logging.INFO)



class RAGASEvaluator:
    """
    RAGAS-inspired Automated RAG Evaluation Engine.
    Measures:
    1. Faithfulness: Ensures the LLM answer does not hallucinate beyond retrieved context.
    2. Context Precision: Evaluates signal-to-noise ratio of retrieved context chunks.
    """

    def __init__(self):
        pass

    def _get_client(self) -> Optional[Any]:
        """Dynamically retrieves GenAI client using active GEMINI_API_KEY."""
        if genai is not None and config.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"RAGASEvaluator GenAI init warning: {e}")
        return None

    def evaluate_response(
        self,
        query: str,
        generation: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates a single RAG response pair.
        Returns score dictionary with faithfulness, context_precision, and detailed reasoning.
        """
        if not generation or not contexts:
            return {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "ragas_score": 0.0,
                "faithfulness_reason": "Missing generation or context",
                "precision_reason": "Empty context provided"
            }

        faithfulness = self._eval_faithfulness(generation, contexts)
        context_precision = self._eval_context_precision(query, contexts)

        ragas_score = round((2 * faithfulness * context_precision) / max(0.001, (faithfulness + context_precision)), 2)

        return {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "ragas_score": min(1.0, ragas_score),
        }

    def _eval_faithfulness(self, generation: str, contexts: List[str]) -> float:
        """Evaluates if generation is grounded in contexts."""
        client = self._get_client()
        if client:
            combined_ctx = "\n".join(f"- {c}" for c in contexts)
            prompt = (
                f"You are a strict RAG Faithfulness Evaluator.\n"
                f"Evaluate if every statement in the Generated Answer is fully supported by the Context.\n"
                f"Context:\n{combined_ctx}\n\n"
                f"Generated Answer:\n{generation}\n\n"
                f"Respond in JSON format with keys:\n"
                f'{{"faithfulness_score": <float between 0.0 and 1.0>, "reason": "<short description>"}}'
            )

            try:
                res = client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt,
                )
                clean_text = res.text.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(clean_text)
                return round(float(parsed.get("faithfulness_score", 0.7)), 2)
            except Exception as e:
                logger.warning(f"Faithfulness eval LLM parse error: {e}. Using heuristic overlap.")

        # Fallback text overlap heuristic
        gen_words = set(generation.lower().split())
        ctx_words = set(" ".join(contexts).lower().split())
        if not gen_words:
            return 0.0
        overlap = len(gen_words.intersection(ctx_words))
        return min(1.0, round(overlap / len(gen_words), 2))

    def _eval_context_precision(self, query: str, contexts: List[str]) -> float:
        """Evaluates precision of retrieved context chunks relative to query."""
        client = self._get_client()
        if client:
            combined_ctx = "\n".join(f"Chunk {i+1}: {c}" for i, c in enumerate(contexts))
            prompt = (
                f"You are a Context Precision Evaluator.\n"
                f"Determine what percentage of the retrieved chunks contain relevant signal to answer the question.\n"
                f"Question: {query}\n\n"
                f"Retrieved Chunks:\n{combined_ctx}\n\n"
                f"Respond in JSON format with keys:\n"
                f'{{"precision_score": <float between 0.0 and 1.0>, "reason": "<short description>"}}'
            )

            try:
                res = client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt,
                )
                clean_text = res.text.strip().replace("```json", "").replace("```", "")
                parsed = json.loads(clean_text)
                return round(float(parsed.get("precision_score", 0.7)), 2)
            except Exception as e:
                logger.warning(f"Context precision LLM parse error: {e}. Using heuristic overlap.")

        # Fallback keyword overlap heuristic
        stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "on", "at", "by", "with"}
        query_words = set(w for w in query.lower().split() if w not in stop_words)
        if not query_words:
            query_words = set(query.lower().split())

        rel_chunks = 0
        for c in contexts:
            c_words = set(c.lower().split())
            if len(query_words.intersection(c_words)) > 0:
                rel_chunks += 1
        return round(rel_chunks / max(1, len(contexts)), 2)

    def run_benchmark_suite(
        self,
        test_cases: List[Dict[str, str]],
        crag_graph: Any
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Executes a test benchmark suite against the CRAG Graph and compiles RAGAS metrics.
        """
        results = []
        total_faith = 0.0
        total_prec = 0.0
        total_latency = 0.0

        for tc in test_cases:
            q = tc["query"]
            start_time = time.time()
            graph_out = crag_graph.run(q)
            latency = round(time.time() - start_time, 2)

            generation = graph_out.get("generation", "")
            graded_docs = graph_out.get("graded_documents", [])
            source_type = graph_out.get("source_type", "")
            contexts = [d["content"] for d in graded_docs]

            eval_res = self.evaluate_response(q, generation, contexts)

            faith = eval_res["faithfulness"]
            prec = eval_res["context_precision"]
            ragas_score = eval_res["ragas_score"]

            total_faith += faith
            total_prec += prec
            total_latency += latency

            results.append({
                "Query": q,
                "Source Type": source_type,
                "Faithfulness": faith,
                "Context Precision": prec,
                "RAGAS Score": ragas_score,
                "Latency (s)": latency,
            })

        df = pd.DataFrame(results)
        num_cases = max(1, len(test_cases))

        degraded = is_degraded_mode()
        active_stubs = get_degraded_components()

        avg_faith = round(total_faith / num_cases, 3)
        avg_prec = round(total_prec / num_cases, 3)
        harmonic_ragas = round((2 * avg_faith * avg_prec) / max(0.001, (avg_faith + avg_prec)), 3)

        summary = {
            "avg_faithfulness": avg_faith,
            "avg_precision": avg_prec,
            "avg_ragas_score": harmonic_ragas,
            "avg_latency_s": round(total_latency / num_cases, 2),
            "degraded_mode": degraded,
            "degraded_components": active_stubs
        }


        if degraded:
            logger.warning(f"RAGAS Benchmark executed in DEGRADED MODE. Active stubs: {', '.join(active_stubs)}")

        return df, summary

