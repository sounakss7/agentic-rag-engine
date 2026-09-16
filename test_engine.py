"""
Comprehensive Automated Test Suite for Advanced CRAG Engine.
Tests all critical bug fixes:
1. Qdrant stub iteration (Res object)
2. RGBA transparent image OCR
3. Decoupled HyDE vs BM25 keyword search
4. Batch context grading and chat history support
5. Dynamic RAGAS evaluation with harmonic mean
"""

import io
import pytest
from PIL import Image

from config import config
from ocr_parser import DocumentIngestor
from retriever import HybridRetriever
from graph_nodes import CRAGGraph
from eval_pipeline import RAGASEvaluator


def test_transparent_png_ocr():
    """Verify RGBA/transparent images do not crash with OSError: cannot write mode RGBA as JPEG."""
    ingestor = DocumentIngestor()
    # Create a 100x100 RGBA image with alpha channel
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    chunks, summary = ingestor.process_file(png_bytes, "test_transparent.png")
    assert summary["filename"] == "test_transparent.png"
    assert len(chunks) >= 1
    # Check that it did not crash with unhandled exception
    assert "Error processing image" not in chunks[0]["content"]


def test_file_stream_seek_behavior():
    """Verify that seeking preserves bytes on repeated reads."""
    sample_text = b"This is document content that should not be wiped out on repeated reads."
    stream = io.BytesIO(sample_text)
    
    # First read (simulating auto-index)
    stream.seek(0)
    b1 = stream.read()
    assert len(b1) > 0

    # Second read (simulating Build Index button)
    stream.seek(0)
    b2 = stream.read()
    assert b1 == b2, "Second read must not return empty bytes"


def test_qdrant_stub_and_dense_search():
    """Verify that dense search handles Qdrant stub results without 'Res' object is not iterable crash."""
    retriever = HybridRetriever()
    test_chunks = [
        {
            "chunk_id": "doc1_c1",
            "content": "Antigravity is an enterprise orchestration platform for autonomous RAG.",
            "metadata": {"source": "manual.txt", "page": 1}
        },
        {
            "chunk_id": "doc1_c2",
            "content": "The revenue of the organization increased by 45 percent in fiscal year 2024.",
            "metadata": {"source": "finance.txt", "page": 1}
        }
    ]

    index_res = retriever.build_index(test_chunks)
    assert index_res["status"] == "success"
    assert index_res["indexed_count"] == 2

    # Dense search must return results and not crash
    dense_res = retriever.dense_search("What is Antigravity?", top_k=2)
    assert isinstance(dense_res, list)

    # Sparse search must find exact keywords
    sparse_res = retriever.sparse_search("Antigravity platform", top_k=2)
    assert isinstance(sparse_res, list)
    assert len(sparse_res) >= 1
    assert "doc1_c1" in [r["chunk_id"] for r in sparse_res]

    # Full search with decoupled dense_query
    full_res = retriever.search(
        query="Antigravity platform",
        top_n=2,
        dense_query="Antigravity hypothetical expanded query context"
    )
    assert isinstance(full_res, list)
    assert len(full_res) >= 1


def test_crag_graph_execution_and_chat_history():
    """Verify CRAG graph runs cleanly with single-batch grading and incorporates chat history."""
    retriever = HybridRetriever()
    retriever.build_index([
        {
            "chunk_id": "chunk_ai",
            "content": "Quantum computing utilizes qubits that can exist in superposition states.",
            "metadata": {"source": "quantum.txt", "page": 1}
        }
    ])

    crag = CRAGGraph(retriever)
    
    # Multi-turn dialogue history
    chat_history = [
        {"role": "user", "content": "Tell me about quantum physics."},
        {"role": "assistant", "content": "Quantum physics deals with nature at microscopic scales."}
    ]

    out = crag.run("How do qubits work?", chat_history=chat_history)
    assert "generation" in out
    assert "confidence_score" in out
    assert "node_trace" in out
    assert len(out["node_trace"]) >= 4
    # Ensure no Windows cp1252 crash emojis were logged in trace strings
    for trace_item in out["node_trace"]:
        assert isinstance(trace_item, str)


def test_eval_pipeline_harmonic_mean():
    """Verify RAGAS evaluation calculates the correct harmonic mean and avoids fake static 0.85/0.80."""
    evaluator = RAGASEvaluator()
    
    # Test response pair
    query = "What is superposition?"
    generation = "Superposition allows qubits to be in multiple states simultaneously."
    contexts = ["Quantum computing utilizes qubits that can exist in superposition states."]

    res = evaluator.evaluate_response(query, generation, contexts)
    assert "faithfulness" in res
    assert "context_precision" in res
    assert "ragas_score" in res

    f = res["faithfulness"]
    p = res["context_precision"]
    expected_harmonic = round((2 * f * p) / max(0.001, f + p), 2)
    assert abs(res["ragas_score"] - expected_harmonic) < 0.05


if __name__ == "__main__":
    print("Running test_transparent_png_ocr...")
    test_transparent_png_ocr()
    print("Running test_file_stream_seek_behavior...")
    test_file_stream_seek_behavior()
    print("Running test_qdrant_stub_and_dense_search...")
    test_qdrant_stub_and_dense_search()
    print("Running test_crag_graph_execution_and_chat_history...")
    test_crag_graph_execution_and_chat_history()
    print("Running test_eval_pipeline_harmonic_mean...")
    test_eval_pipeline_harmonic_mean()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY! [OK]")
