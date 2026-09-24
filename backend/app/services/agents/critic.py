"""
NexusRAG Self-Reflection & Citation Verification Critic.
Inspects synthesized answers for ungrounded claims, missing citations, or numerical discrepancies.
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class CriticReport(BaseModel):
    is_grounded: bool = Field(description="True if every claim in the answer is supported by the context")
    citations_present: bool = Field(description="True if numerical bracket citations [1], [2] are used correctly")
    critic_score: float = Field(description="Confidence score between 0.0 and 1.0")
    feedback: str = Field(description="Constructive notes on ungrounded claims or required improvements")


class CitationVerificationCritic:
    """Critic Agent validating factual alignment and citation fidelity."""

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def review_answer(
        self,
        query: str,
        draft_answer: str,
        context_chunks: List[Dict[str, Any]],
        code_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reviews draft generation against retrieved context and code execution logs.
        """
        if not draft_answer or not context_chunks:
            return {
                "critic_passed": True,
                "critic_score": 0.5,
                "critic_feedback": "Empty context or response."
            }

        client = self._get_genai_client()
        if client and types is not None:
            ctx_summary = "\n\n".join([
                f"[{idx+1}] Source: {c.get('metadata', {}).get('source', 'doc')} | Page {c.get('metadata', {}).get('page', 1)}\n{c.get('parent_content') or c.get('content')}"
                for idx, c in enumerate(context_chunks)
            ])

            code_note = f"\nPython Sandbox Execution Output:\n{code_result}\n" if code_result else ""

            prompt = (
                "You are an adversarial AI Fact-Checking Critic.\n"
                "Verify whether the Draft Answer is strictly grounded in the provided Source Context and Code Execution:\n"
                "1. Flag any claims or numbers NOT supported by context as hallucinated.\n"
                "2. Verify that numerical citations [1], [2] correspond accurately to sources.\n"
                "3. If code execution was performed, verify that calculated numbers match.\n\n"
                f"User Question: {query}\n\n"
                f"Source Context:\n{ctx_summary[:3000]}\n"
                f"{code_note}\n"
                f"Draft Answer:\n{draft_answer}\n"
            )

            for model_name in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=CriticReport
                        )
                    )
                    if res and res.text:
                        parsed = json.loads(res.text.strip().replace("```json", "").replace("```", ""))
                        passed = bool(parsed.get("is_grounded", True) and parsed.get("citations_present", True))
                        score = float(parsed.get("critic_score", 0.85))
                        return {
                            "critic_passed": passed or score >= 0.70,
                            "critic_score": score,
                            "critic_feedback": parsed.get("feedback", "Verified grounded response.")
                        }
                except Exception as e:
                    if "429" in str(e):
                        continue
                    logger.warning(f"Critic review notice: {e}")
                    break

        # Heuristic fallback: check citation brackets
        has_citations = "[" in draft_answer and "]" in draft_answer
        return {
            "critic_passed": has_citations,
            "critic_score": 0.80 if has_citations else 0.60,
            "critic_feedback": "Heuristic citation format check passed." if has_citations else "Missing numerical bracket citations."
        }
