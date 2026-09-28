"""
NexusRAG Multi-Agent State Definition.
TypedDict schema tracking execution traces, sub-task plans, code sandbox results, and critic feedback.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class SubTask(TypedDict):
    task_id: str
    description: str
    target_specialist: str  # "vector_graph", "web_search", "code_interpreter"
    query: str
    status: str  # "pending", "completed", "failed"
    result: Optional[str]


class NexusAgentState(TypedDict):
    """Global state flowing through the NexusRAG LangGraph multi-agent network."""
    # User Request Context
    query: str
    chat_history: List[Dict[str, Any]]

    # Planning & Decomposition
    hyde_expansion: str
    is_quantitative: bool
    is_document_summary: Optional[bool]
    enabled_specialists: Optional[List[str]]
    plan: List[Dict[str, Any]]
    current_step: int

    # Retrieval Assets
    retrieved_documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]
    web_documents: List[Dict[str, Any]]

    # Computational Tools
    code_generated: Optional[str]
    code_execution_result: Optional[str]

    # Synthesis & Verification
    draft_answer: str
    final_answer: str
    confidence_score: float
    source_type: str
    citations: List[Dict[str, Any]]

    # Critic & Self-Reflection
    critic_passed: bool
    critic_feedback: str
    refinement_count: int

    # Telemetry
    node_trace: List[str]
