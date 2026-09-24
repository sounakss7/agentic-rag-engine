"""
NexusRAG Comprehensive Automated Verification Test Suite.
Tests all advanced subsystems:
1. Layout-aware table extraction
2. Parent-Child Hierarchical Chunking
3. Knowledge Graph (GraphRAG) construction and subgraph search
4. Python Code Interpreter Sandbox
5. Multi-Agent LangGraph Orchestration & Critic loop
6. FastAPI REST & health endpoints
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import settings
from backend.app.services.ingestion.parser import LayoutAwareParser
from backend.app.services.ingestion.chunker import HierarchicalChunker
from backend.app.services.retrieval.graph_store import GraphStoreService
from backend.app.services.retrieval.hybrid import HybridRetrievalService
from backend.app.services.agents.code_executor import CodeExecutionSandbox
from backend.app.services.agents.orchestrator import NexusOrchestrator
from backend.app.main import app


def test_layout_aware_table_parsing():
    """Verify that tabular datasets are converted into structured Markdown tables."""
    parser = LayoutAwareParser()
    csv_bytes = b"Quarter,Revenue,Growth\nQ1,$10.5B,12%\nQ2,$12.0B,15%\nQ3,$14.2B,18%\n"
    pages = parser.parse_document(csv_bytes, "financials.csv")

    assert len(pages) == 1
    assert "Quarter" in pages[0]["text"]
    assert "| Q1 | $10.5B | 12% |" in pages[0]["text"]
    assert len(pages[0]["tables"]) >= 1


def test_parent_child_hierarchical_chunking():
    """Verify parent-child chunk linkage and preservation of parent blocks."""
    chunker = HierarchicalChunker(parent_chunk_size=400, child_chunk_size=100, chunk_overlap=20)
    pages_data = [{
        "page": 1,
        "text": (
            "NexusRAG is an enterprise research system that integrates Knowledge Graphs with vector search. "
            "It incorporates an autonomous planner that decomposes complex queries into parallel tasks. "
            "Furthermore, it includes a Python Code Interpreter sandbox to eliminate math hallucinations. "
            "This ensures 100 percent numerical precision across financial reports."
        ),
        "tables": [],
        "ocr_used": False,
        "source": "overview.txt"
    }]

    children, parent_map = chunker.process_document(pages_data, "overview.txt")

    assert len(children) >= 2
    assert len(parent_map) >= 1

    # Verify that child chunks point to their valid parent
    for child in children:
        p_id = child["parent_id"]
        assert p_id in parent_map
        assert child["content"] in parent_map[p_id]["content"]


def test_knowledge_graph_builder_and_subgraph_search():
    """Verify Knowledge Graph triple extraction and 1-hop subgraph discovery."""
    graph_store = GraphStoreService()
    parent_sections = {
        "sec_1": {
            "content": "Microsoft reported total cloud revenue of $35 Billion in fiscal year 2024. Azure increased by 29 percent.",
            "metadata": {"source": "sec_filing.txt", "page": 2}
        },
        "sec_2": {
            "content": "OpenAI partnered with Microsoft to deploy advanced models on Azure infrastructure.",
            "metadata": {"source": "press_release.txt", "page": 1}
        }
    }

    triples_count = graph_store.build_graph_from_sections(parent_sections)
    assert graph_store.graph.number_of_nodes() >= 2
    assert graph_store.graph.number_of_edges() >= 1

    # Search subgraph for relational query
    subgraph_hits = graph_store.search_subgraph("What was Microsoft Azure revenue?")
    assert isinstance(subgraph_hits, list)

    # Visualization data format test
    vis = graph_store.get_visualization_data(max_nodes=10)
    assert "nodes" in vis
    assert "edges" in vis


def test_python_code_interpreter_sandbox():
    """Verify deterministic Python execution in controlled sandbox."""
    sandbox = CodeExecutionSandbox()
    test_code = """
import pandas as pd
data = {"Quarter": ["Q1", "Q2", "Q3"], "Revenue": [10.5, 12.0, 14.2]}
df = pd.DataFrame(data)
diff = df['Revenue'].iloc[-1] - df['Revenue'].iloc[0]
print(f"Revenue growth: ${diff:.2f} Billion")
"""
    success, output = sandbox.execute_code(test_code)
    assert success is True
    assert "Revenue growth: $3.70 Billion" in output


def test_nexus_multi_agent_orchestrator():
    """Verify full multi-agent LangGraph workflow execution with chat history."""
    hybrid_service = HybridRetrievalService()
    test_pages = [{
        "page": 1,
        "text": "Antigravity is an enterprise orchestration platform for autonomous GraphRAG. Revenue reached $100M.",
        "tables": [],
        "ocr_used": False,
        "source": "antigravity.txt"
    }]
    children, parents = HierarchicalChunker().process_document(test_pages, "antigravity.txt")
    hybrid_service.build_index(children, parents)

    orchestrator = NexusOrchestrator(hybrid_service)
    chat_history = [{"role": "user", "content": "What is Antigravity?"}]

    res = orchestrator.run("What was the revenue reached by Antigravity?", chat_history=chat_history)
    assert "final_answer" in res
    assert "confidence_score" in res
    assert "node_trace" in res
    assert len(res["node_trace"]) >= 4


def test_fastapi_rest_endpoints():
    """Verify FastAPI REST API functionality."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Root endpoint
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "documentation" in res_root.json()

    # Documents listing
    res_docs = client.get("/api/v1/documents")
    assert res_docs.status_code == 200
    assert "total_child_chunks" in res_docs.json()


if __name__ == "__main__":
    print("Running test_layout_aware_table_parsing...")
    test_layout_aware_table_parsing()
    print("Running test_parent_child_hierarchical_chunking...")
    test_parent_child_hierarchical_chunking()
    print("Running test_knowledge_graph_builder_and_subgraph_search...")
    test_knowledge_graph_builder_and_subgraph_search()
    print("Running test_python_code_interpreter_sandbox...")
    test_python_code_interpreter_sandbox()
    print("Running test_nexus_multi_agent_orchestrator...")
    test_nexus_multi_agent_orchestrator()
    print("Running test_fastapi_rest_endpoints...")
    test_fastapi_rest_endpoints()
    print("\nALL NEXUSRAG AUTOMATED TESTS PASSED SUCCESSFULLY! [OK]")
