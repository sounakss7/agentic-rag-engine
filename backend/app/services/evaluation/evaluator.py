"""
NexusRAG Production Evaluation & Benchmarking Pipeline.
Measures Faithfulness, Context Precision, True Harmonic Mean (F1), and Latency.
"""

import time
import json
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
except ImportError:
    genai = None


class NexusEvaluator:
    """Production RAGAS evaluator for quantitative quality assurance."""

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def evaluate_response(
        self,
        query: str,
        generation: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates Faithfulness, Context Precision, and the Harmonic Mean Score.
        """
        if not generation or not contexts:
            return {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "ragas_score": 0.0,
            }

        faithfulness = self._eval_faithfulness(generation, contexts)
        precision = self._eval_context_precision(query, contexts)

        denom = faithfulness + precision
        ragas_f1 = round((2 * faithfulness * precision) / max(0.001, denom), 2)

        return {
            "faithfulness": faithfulness,
            "context_precision": precision,
            "ragas_score": min(1.0, ragas_f1)
        }

    def _eval_faithfulness(self, generation: str, contexts: List[str]) -> float:
        client = self._get_genai_client()
        if client:
            ctx_text = "\n".join([f"- {c[:400]}" for c in contexts[:5]])
            prompt = (
                "You are an evaluator assessing RAG Faithfulness.\n"
                "Determine the fraction (0.0 to 1.0) of statements in the Generated Answer that are supported by the Context.\n"
                f"Context:\n{ctx_text}\n\n"
                f"Generated Answer:\n{generation}\n\n"
                "Return JSON with key: {\"faithfulness\": <float between 0.0 and 1.0>}"
            )
            for model in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(model=model, contents=prompt)
                    if res and res.text:
                        clean = res.text.strip().replace("```json", "").replace("```", "")
                        data = json.loads(clean)
                        return float(data.get("faithfulness", 0.85))
                except Exception as e:
                    if "429" in str(e):
                        continue
                    break

        # Lexical fallback
        gen_words = set(generation.lower().split())
        ctx_words = set(" ".join(contexts).lower().split())
        if not gen_words:
            return 0.0
        overlap = len(gen_words.intersection(ctx_words))
        return min(1.0, round(overlap / len(gen_words), 2))

    def _eval_context_precision(self, query: str, contexts: List[str]) -> float:
        client = self._get_genai_client()
        if client:
            ctx_text = "\n".join([f"Chunk {i+1}: {c[:300]}" for i, c in enumerate(contexts[:5])])
            prompt = (
                "You are an evaluator assessing Context Precision.\n"
                "Determine the proportion (0.0 to 1.0) of retrieved chunks that contain relevant factual signal for the question.\n"
                f"Question: {query}\n\n"
                f"Retrieved Chunks:\n{ctx_text}\n\n"
                "Return JSON with key: {\"precision\": <float between 0.0 and 1.0>}"
            )
            for model in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(model=model, contents=prompt)
                    if res and res.text:
                        clean = res.text.strip().replace("```json", "").replace("```", "")
                        data = json.loads(clean)
                        return float(data.get("precision", 0.80))
                except Exception as e:
                    if "429" in str(e):
                        continue
                    break

        # Lexical fallback
        q_words = set(query.lower().split())
        if not q_words or not contexts:
            return 0.0
        relevant_count = sum(1 for c in contexts if any(w in c.lower() for w in q_words))
        return min(1.0, round(relevant_count / len(contexts), 2))
