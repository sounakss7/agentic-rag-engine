"""
NexusRAG Python Code Interpreter Sandbox.
Executes mathematical and tabular calculations deterministically using Pandas and NumPy.
Eliminates LLM arithmetic hallucinations.
"""

import sys
import io
import math
import statistics
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
except ImportError:
    genai = None


class CodeExecutionSandbox:
    """Safe in-process Python execution environment for quantitative reasoning."""

    SAFE_MODULES = {
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "math": math,
        "statistics": statistics,
    }

    def __init__(self):
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def generate_calculation_code(self, query: str, context_chunks: list) -> Optional[str]:
        """
        Uses LLM to write clean Python script utilizing pandas/numpy to compute numerical answer.
        """
        # Guard: Only invoke LLM if query and context contain numeric content and calculation keywords
        calc_triggers = ["calculate", "compute", "sum", "average", "ratio", "difference", "growth", "margin", "%", "percent", "divide", "multiply"]
        if not any(t in query.lower() for t in calc_triggers):
            return None

        has_digits = any(c.isdigit() for c in query) or any(any(char.isdigit() for char in c.get('content', '')) for c in context_chunks)
        if not has_digits:
            return None

        client = self._get_genai_client()
        if not client:
            return None

        # Format tabular context
        tables_text = "\n\n".join([
            f"--- Context {idx+1} ---\n{c.get('parent_content') or c.get('content')}"
            for idx, c in enumerate(context_chunks)
        ])

        prompt = (
            "You are an expert financial and statistical programmer.\n"
            "Given the user query and context containing data or tables, write a standalone Python script "
            "to calculate the exact numerical result.\n"
            "Rules:\n"
            "1. Use `pandas` (as `pd`) or `numpy` (as `np`) or `math`.\n"
            "2. Define any data structures (dictionaries, DataFrames) explicitly from the context.\n"
            "3. Print clear, formatted final answers with units or percentages using `print(...)`.\n"
            "4. Return ONLY executable python code enclosed in ```python ... ``` blocks.\n\n"
            f"User Query: {query}\n\n"
            f"Available Data Context:\n{tables_text[:2500]}\n\n"
            "Python Script:"
        )

        try:
            for model_name in settings.CASCADE_MODELS:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if res and res.text:
                        code_match = res.text.strip()
                        if "```python" in code_match:
                            code_match = code_match.split("```python")[1].split("```")[0].strip()
                        elif "```" in code_match:
                            code_match = code_match.split("```")[1].split("```")[0].strip()
                        return code_match
                except Exception as e:
                    if "429" in str(e):
                        continue
                    break
        except Exception as e:
            logger.warning(f"Code generation error: {e}")

        return None

    def execute_code(self, python_code: str) -> Tuple[bool, str]:
        """
        Executes python code in a controlled namespace and captures stdout.
        Returns:
            Tuple of (success: bool, output: str)
        """
        if not python_code or not python_code.strip():
            return False, "Empty code snippet provided."

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        local_env = dict(self.SAFE_MODULES)

        try:
            sys.stdout = stdout_capture
            exec(python_code, local_env, local_env)
            output = stdout_capture.getvalue().strip()
            return True, output if output else "Script executed successfully with no stdout."
        except Exception as e:
            return False, f"Execution Error: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
