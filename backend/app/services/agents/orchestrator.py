"""
NexusRAG LangGraph Multi-Agent Orchestrator.
Compiles the dynamic, self-correcting StateGraph workflow and provides real-time streaming event hooks.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from langgraph.graph import StateGraph, START, END

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.retrieval.hybrid import HybridRetrievalService
from backend.app.services.agents.state import NexusAgentState
from backend.app.services.agents.planner import MasterPlannerAgent
from backend.app.services.agents.researchers import ResearchSpecialists
from backend.app.services.agents.code_executor import CodeExecutionSandbox
from backend.app.services.agents.critic import CitationVerificationCritic
from backend.app.services.agents.synthesizer import AnswerSynthesizer


class NexusOrchestrator:
    """Compiled LangGraph orchestrator governing autonomous research, code execution, and verification."""

    def __init__(self, hybrid_service: Optional[HybridRetrievalService] = None):
        self.hybrid_service = hybrid_service or HybridRetrievalService()
        self.planner = MasterPlannerAgent()
        self.researchers = ResearchSpecialists(self.hybrid_service)
        self.code_sandbox = CodeExecutionSandbox()
        self.critic = CitationVerificationCritic()
        self.synthesizer = AnswerSynthesizer()

        self.graph = self._build_graph()

    # ==================== NODE DEFINITIONS ====================

    def planner_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 1: Master Planner Agent."""
        query = state["query"]
        history = state.get("chat_history", [])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 1: Master Planner] Analyzing intent, decomposing query & expanding HyDE...")

        plan_meta = self.planner.plan_query(query, history)
        sub_tasks = plan_meta.get("sub_tasks", [query])
        is_quant = plan_meta.get("is_quantitative", False)
        hyde = plan_meta.get("hyde_expansion", query)

        trace.append(f"   [+] Decomposed into {len(sub_tasks)} sub-tasks. Quantitative Mode: {is_quant}")
        return {
            "is_quantitative": is_quant,
            "plan": [{"task_id": f"t_{i}", "query": q} for i, q in enumerate(sub_tasks)],
            "hyde_expansion": hyde,
            "node_trace": trace
        }

    def vector_graph_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 2: Specialist Vector + Graph Search."""
        plan = state.get("plan", [])
        sub_tasks = [t["query"] for t in plan] if plan else [state["query"]]
        hyde = state.get("hyde_expansion", state["query"])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 2: Vector & Graph Specialist] Querying Qdrant Dense, BM25 & Knowledge Graph...")

        retrieved = self.researchers.execute_vector_graph_search(sub_tasks, hyde)
        graph_facts = [r for r in retrieved if r.get("metadata", {}).get("is_graph", False)]
        doc_chunks = [r for r in retrieved if not r.get("metadata", {}).get("is_graph", False)]

        trace.append(f"   [+] Fused {len(doc_chunks)} document passages and {len(graph_facts)} Knowledge Graph facts.")
        return {
            "retrieved_documents": doc_chunks,
            "graph_facts": graph_facts,
            "node_trace": trace
        }

    def context_grader_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 3: Batched Context Grader Judge."""
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 3: Context Grader] Evaluating chunk relevance in single batched judge call...")

        graded, fallback_needed = self.researchers.grade_context_batch(query, docs)
        trace.append(f"   [+] Passed {len(graded)}/{len(docs)} chunks. Fallback Required: {fallback_needed}")
        return {
            "graded_documents": graded,
            "node_trace": trace
        }

    def web_fallback_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 4: Tavily Deep Web Researcher."""
        query = state["query"]
        trace = list(state.get("node_trace", []))
        trace.append("[Node 4: Deep Web Researcher] Corrective routing triggered; querying live web...")

        web_hits = self.researchers.execute_web_search(query)
        trace.append(f"   [+] Retrieved {len(web_hits)} verified live web results.")
        return {
            "web_documents": web_hits,
            "graded_documents": web_hits if not state.get("graded_documents") else state["graded_documents"] + web_hits,
            "source_type": "Corrected via Live Web Grounding (Tavily)",
            "node_trace": trace
        }

    def code_executor_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 5: Python Code Sandbox Interpreter."""
        query = state["query"]
        graded = state.get("graded_documents", [])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 5: Python Code Interpreter] Tabular/numerical query detected; generating sandbox script...")

        python_code = self.code_sandbox.generate_calculation_code(query, graded)
        code_out = None

        if python_code:
            success, output = self.code_sandbox.execute_code(python_code)
            code_out = output
            trace.append(f"   [+] Executed deterministic Python code: Success={success}. Output preview: {output[:100]}...")
        else:
            trace.append("   [!] Code generation skipped or not required.")

        return {
            "code_generated": python_code,
            "code_execution_result": code_out,
            "node_trace": trace
        }

    def synthesizer_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 6: Final Context Fusion & Answer Synthesizer."""
        query = state["query"]
        history = state.get("chat_history", [])
        docs = state.get("graded_documents", [])
        graph_facts = state.get("graph_facts", [])
        code_result = state.get("code_execution_result")
        feedback = state.get("critic_feedback")
        trace = list(state.get("node_trace", []))
        trace.append("[Node 6: Answer Synthesizer] Synthesizing comprehensive cited response...")

        answer, citations, conf = self.synthesizer.synthesize_response(
            query=query,
            chat_history=history,
            context_chunks=docs,
            graph_facts=graph_facts,
            code_result=code_result,
            critic_feedback=feedback
        )

        src_type = state.get("source_type") or ("Retrieved from Qdrant & Knowledge Graph" if docs else "Unresolved Context")
        trace.append(f"   [+] Synthesis complete. Confidence: {conf*100:.0f}%. Source: {src_type}")
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "confidence_score": conf,
            "citations": citations,
            "source_type": src_type,
            "node_trace": trace
        }

    def critic_node(self, state: NexusAgentState) -> Dict[str, Any]:
        """Node 7: Fact-Checking & Citation Verification Critic."""
        query = state["query"]
        draft = state.get("draft_answer", "")
        docs = state.get("graded_documents", [])
        code_out = state.get("code_execution_result")
        count = state.get("refinement_count", 0)
        trace = list(state.get("node_trace", []))
        trace.append("[Node 7: Verification Critic] Verifying factual grounding and citation precision...")

        report = self.critic.review_answer(query, draft, docs, code_out)
        passed = report.get("critic_passed", True)
        feedback = report.get("critic_feedback", "")
        trace.append(f"   [+] Verification Result: Passed={passed}, Score={report.get('critic_score')}. Feedback: {feedback}")

        return {
            "critic_passed": passed,
            "critic_feedback": feedback,
            "refinement_count": count + 1,
            "node_trace": trace
        }

    # ==================== ROUTING LOGIC ====================

    @staticmethod
    def route_after_grading(state: NexusAgentState) -> str:
        graded = state.get("graded_documents", [])
        if not graded:
            return "web_fallback"
        if state.get("is_quantitative", False):
            return "code_executor"
        return "synthesizer"

    @staticmethod
    def route_after_web(state: NexusAgentState) -> str:
        if state.get("is_quantitative", False):
            return "code_executor"
        return "synthesizer"

    @staticmethod
    def route_after_critic(state: NexusAgentState) -> str:
        if not state.get("critic_passed", True) and state.get("refinement_count", 0) < 2:
            return "synthesizer"
        return END

    # ==================== GRAPH BUILDER ====================

    def _build_graph(self) -> Any:
        workflow = StateGraph(NexusAgentState)

        workflow.add_node("planner", self.planner_node)
        workflow.add_node("vector_graph", self.vector_graph_node)
        workflow.add_node("context_grader", self.context_grader_node)
        workflow.add_node("web_fallback", self.web_fallback_node)
        workflow.add_node("code_executor", self.code_executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("critic", self.critic_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "vector_graph")
        workflow.add_edge("vector_graph", "context_grader")

        workflow.add_conditional_edges(
            "context_grader",
            self.route_after_grading,
            {
                "web_fallback": "web_fallback",
                "code_executor": "code_executor",
                "synthesizer": "synthesizer",
            }
        )

        workflow.add_conditional_edges(
            "web_fallback",
            self.route_after_web,
            {
                "code_executor": "code_executor",
                "synthesizer": "synthesizer",
            }
        )

        workflow.add_edge("code_executor", "synthesizer")
        workflow.add_edge("synthesizer", "critic")

        workflow.add_conditional_edges(
            "critic",
            self.route_after_critic,
            {
                "synthesizer": "synthesizer",
                END: END
            }
        )

        return workflow.compile()

    def run(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Executes the full NexusRAG autonomous multi-agent workflow."""
        initial_state: NexusAgentState = {
            "query": query,
            "chat_history": chat_history or [],
            "hyde_expansion": "",
            "is_quantitative": False,
            "plan": [],
            "current_step": 0,
            "retrieved_documents": [],
            "graded_documents": [],
            "graph_facts": [],
            "web_documents": [],
            "code_generated": None,
            "code_execution_result": None,
            "draft_answer": "",
            "final_answer": "",
            "confidence_score": 0.0,
            "source_type": "Retrieved from Qdrant Vector Store & Knowledge Graph",
            "citations": [],
            "critic_passed": True,
            "critic_feedback": "",
            "refinement_count": 0,
            "node_trace": []
        }

        return self.graph.invoke(initial_state)
