"""
NexusRAG Master Planner Agent.
Performs autonomous query decomposition, HyDE semantic expansion, and quantitative task detection.
"""

import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class PlanDecomposition(BaseModel):
    is_quantitative: bool = Field(description="True ONLY if query explicitly asks to compute, calculate, or perform mathematical operations")
    sub_tasks: List[str] = Field(description="List of decomposed sub-questions or focused queries to retrieve complete context")
    hyde_expansion: str = Field(description="Concise hypothetical technical paragraph answering the query directly for vector expansion")


class MasterPlannerAgent:
    """Master Planner Agent orchestrating task decomposition and intent analysis."""

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def plan_query(self, query: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decomposes user query and produces an execution roadmap.
        Includes zero-latency fast-track detection for document summaries.
        """
        q_lower = query.lower().strip()

        # 1. Fast-track for document summarization or overview requests
        summary_triggers = [
            "summary", "summarize", "overview", "what is this document", "what is the document",
            "from the above document", "from this document", "tell me about the document",
            "explain the document", "briefly describe the document", "main points"
        ]
        is_doc_summary = any(tr in q_lower for tr in summary_triggers)
        if is_doc_summary:
            return {
                "is_quantitative": False,
                "is_document_summary": True,
                "sub_tasks": [query, "Document executive summary, introduction, and principal findings"],
                "hyde_expansion": query,
            }

        # 2. Strict quantitative calculation check
        math_triggers = [
            "calculate", "compute", "difference between", "percentage increase",
            "growth rate", "sum of", "average of", "ratio of", "multiply", "divide"
        ]
        has_math_words = any(m in q_lower for m in math_triggers)
        has_digits = any(c.isdigit() for c in query)
        is_strictly_quant = has_math_words and (has_digits or "calculate" in q_lower or "compute" in q_lower)

        client = self._get_genai_client()
        if not client:
            return {
                "is_quantitative": is_strictly_quant,
                "is_document_summary": False,
                "sub_tasks": [query],
                "hyde_expansion": query,
            }

        history_context = ""
        if chat_history:
            recent = chat_history[-3:]
            history_context = "Prior Conversation:\n" + "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent]) + "\n\n"

        prompt = (
            "You are a Principal AI Research Planner.\n"
            "Analyze the user's question and produce a structured execution plan:\n"
            "1. Detect if the question explicitly requires mathematical calculation (`is_quantitative`). Default to false unless math is explicitly required.\n"
            "2. Decompose comparative or multi-part questions into 1-2 targeted sub-queries (`sub_tasks`).\n"
            "3. Generate a hypothetical factual sentence answering the question (`hyde_expansion`) to maximize vector retrieval recall.\n\n"
            f"{history_context}"
            f"User Question: {query}\n"
        )

        for model_name in settings.CASCADE_MODELS:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PlanDecomposition,
                        temperature=0.1
                    ) if types else None
                )
                if res and res.text:
                    parsed = json.loads(res.text.strip().replace("```json", "").replace("```", ""))
                    return {
                        "is_quantitative": bool(parsed.get("is_quantitative", False)) and is_strictly_quant,
                        "is_document_summary": False,
                        "sub_tasks": parsed.get("sub_tasks", [query]) or [query],
                        "hyde_expansion": parsed.get("hyde_expansion", query) or query,
                    }
            except Exception as e:
                if "429" in str(e):
                    continue
                logger.warning(f"Planner LLM notice: {e}")
                break

        # Fast heuristic fallback
        return {
            "is_quantitative": is_strictly_quant,
            "is_document_summary": False,
            "sub_tasks": [query],
            "hyde_expansion": query,
        }
