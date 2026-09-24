"""
NexusRAG Final Context Fusion & Answer Synthesizer.
Synthesizes verified answers incorporating tabular code outputs, Graph facts, and numerical citations.
"""

from typing import List, Dict, Any, Tuple, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
except ImportError:
    genai = None


class AnswerSynthesizer:
    """Synthesizes factual, cited responses from multi-source research assets."""

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def synthesize_response(
        self,
        query: str,
        chat_history: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
        graph_facts: List[Dict[str, Any]],
        code_result: Optional[str] = None,
        critic_feedback: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Synthesizes grounded final answer with citations and confidence metrics.
        Returns:
            Tuple of (answer_text, citations_list, confidence_score)
        """
        citations = []
        context_blocks = []
        total_score = 0.0

        for idx, doc in enumerate(context_chunks):
            src = doc.get("metadata", {}).get("source", f"Document-{idx+1}")
            page = doc.get("metadata", {}).get("page", 1)
            content = doc.get("parent_content") or doc.get("content", "")
            context_blocks.append(f"--- [Source {idx+1}: {src} (Page {page})] ---\n{content}\n")
            citations.append({
                "citation_index": idx + 1,
                "source": src,
                "page": page,
                "snippet": content[:300],
                "url": doc.get("metadata", {}).get("url", "")
            })
            total_score += doc.get("relevance_score", 0.8)

        # Graph facts block
        graph_str = ""
        if graph_facts:
            graph_lines = [f"- {gf['content']}" for gf in graph_facts]
            graph_str = "\n[Knowledge Graph Relational Context]:\n" + "\n".join(graph_lines) + "\n\n"

        # Code execution block
        code_str = ""
        if code_result:
            code_str = f"\n[Deterministic Python Sandbox Calculation Results]:\n{code_result}\n\n"

        history_str = ""
        if chat_history:
            turns = chat_history[-3:]
            history_str = "Prior Dialogue:\n" + "\n".join([f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in turns]) + "\n\n"

        refinement_note = f"\nCRITIC FEEDBACK TO ADDRESS:\n{critic_feedback}\n" if critic_feedback else ""

        client = self._get_genai_client()
        confidence = round(total_score / max(1, len(context_chunks)), 2) if context_chunks else 0.0

        if client and context_blocks:
            prompt = (
                "You are NexusRAG, an enterprise principal research intelligence system.\n"
                "Provide a comprehensive, authoritative, and strictly grounded answer to the user's question.\n"
                "Requirements:\n"
                "1. Cite sources with explicit numerical brackets matching context chunks, e.g. [1], [2].\n"
                "2. If Python sandbox calculation results are provided, cite the exact numbers from the sandbox.\n"
                "3. If tables or financial metrics are present, preserve clean formatting.\n"
                "4. Do NOT invent claims outside the context.\n\n"
                f"{history_str}"
                f"User Question: {query}\n\n"
                f"Context Chunks:\n{''.join(context_blocks)}\n"
                f"{graph_str}"
                f"{code_str}"
                f"{refinement_note}"
                "Comprehensive Answer:"
            )

            for model_name in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if res and res.text:
                        return res.text.strip(), citations, confidence
                except Exception as e:
                    if "429" in str(e):
                        continue
                    logger.warning(f"Synthesis error on {model_name}: {e}")
                    break

        elif not context_blocks:
            return (
                f"No verified information regarding '{query}' was found in the indexed documents or external fallback. "
                "Please verify your query or upload relevant reference files.",
                [],
                0.0
            )

        # Heuristic fallback if LLM is unavailable
        fallback_text = (
            f"### Synthesized Answer for: {query}\n\n"
            f"Based on retrieved documentation:\n\n"
            + "".join(context_blocks[:2])
            + (f"\n\n**Calculations:**\n```\n{code_result}\n```\n" if code_result else "")
        )
        return fallback_text, citations, confidence
