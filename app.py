"""
NexusRAG Enterprise Control Center & Interactive Multi-Agent Research Interface.
Provides interactive Deep-Research chat, sub-task visualization, Python code sandbox output,
Knowledge Graph exploration, Parent-Child chunk inspection, and RAGAS benchmarking.
"""

import os
import sys
import time
import pandas as pd
import streamlit as st

# Add workspace to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.core.config import settings, get_degraded_status
from backend.app.core.logging import logger
from backend.app.services.ingestion.parser import LayoutAwareParser
from backend.app.services.ingestion.chunker import HierarchicalChunker
from backend.app.services.retrieval.hybrid import HybridRetrievalService
from backend.app.services.agents.orchestrator import NexusOrchestrator
from backend.app.services.evaluation.evaluator import NexusEvaluator

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="NexusRAG | Autonomous Deep-Research & Graph-RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Glassmorphic Dark Styling
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #090d16 0%, #0d1527 100%); }
    .stApp { color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    
    .nexus-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .badge-qdrant {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-web {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-code {
        background: linear-gradient(90deg, #8b5cf6 0%, #6d28d9 100%);
        color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
        color: white; font-weight: 600; border: none; border-radius: 8px;
        padding: 0.5rem 1.25rem; transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ==================== INITIALIZE STATE & SERVICES ====================
if "parser" not in st.session_state:
    st.session_state.parser = LayoutAwareParser()
if "chunker" not in st.session_state:
    st.session_state.chunker = HierarchicalChunker()
if "hybrid_service" not in st.session_state:
    st.session_state.hybrid_service = HybridRetrievalService()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = NexusOrchestrator(st.session_state.hybrid_service)
if "evaluator" not in st.session_state:
    st.session_state.evaluator = NexusEvaluator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_docs_meta" not in st.session_state:
    st.session_state.indexed_docs_meta = []


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚡ NexusRAG")
    st.markdown("*Autonomous Deep-Research & Graph-RAG*")
    st.markdown("---")

    # API Key & Configuration Expander
    with st.expander("🔑 API Keys & Settings", expanded=False):
        gemini_input = st.text_input("Gemini API Key", value=settings.active_gemini_key or "", type="password")
        if gemini_input:
            settings.GEMINI_API_KEY = gemini_input

        tavily_input = st.text_input("Tavily API Key", value=settings.active_tavily_key or "", type="password")
        if tavily_input:
            settings.TAVILY_API_KEY = tavily_input

        qdrant_url_in = st.text_input("Qdrant URL", value=settings.active_qdrant_url or "")
        if qdrant_url_in:
            settings.QDRANT_URL = qdrant_url_in

        qdrant_key_in = st.text_input("Qdrant API Key", value=settings.active_qdrant_key or "", type="password")
        if qdrant_key_in:
            settings.QDRANT_API_KEY = qdrant_key_in

    # Ingestion Section
    st.markdown("### 📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Reference Documents",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents"):
            with st.spinner("Executing Layout-Aware Parsing, Hierarchical Chunking & Graph Extraction..."):
                all_child_chunks = []
                all_parents = {}

                for uf in uploaded_files:
                    uf.seek(0)
                    file_bytes = uf.read()
                    filename = uf.name

                    pages_data = st.session_state.parser.parse_document(file_bytes, filename)
                    child_chunks, parent_map = st.session_state.chunker.process_document(pages_data, filename)

                    all_child_chunks.extend(child_chunks)
                    all_parents.update(parent_map)
                    st.session_state.indexed_docs_meta.append({
                        "filename": filename,
                        "pages": len(pages_data),
                        "chunks": len(child_chunks)
                    })

                index_res = st.session_state.hybrid_service.build_index(all_child_chunks, all_parents)
                st.success(f"Indexed {index_res['indexed_chunks']} atomic chunks & {index_res['graph_triples']} Graph relations!")

    # System Status Metrics
    st.markdown("---")
    st.markdown("### 📊 Engine Status")
    status = settings.key_status()
    st.markdown(f"**Gemini API:** {'🟢 Active' if status['GEMINI_API_KEY'] else '🔴 Missing'}")
    st.markdown(f"**Qdrant Cloud:** {'🟢 Active' if status['QDRANT_URL'] else '🟡 Local Memory'}")
    st.markdown(f"**Tavily Web Search:** {'🟢 Active' if status['TAVILY_API_KEY'] else '🔴 Missing'}")
    st.markdown(f"**Knowledge Graph:** {st.session_state.hybrid_service.graph_store.graph.number_of_nodes()} Nodes | {st.session_state.hybrid_service.graph_store.graph.number_of_edges()} Edges")

    degraded = get_degraded_status()
    if degraded:
        st.warning(f"Active Fallbacks: {', '.join(degraded.keys())}")

    if st.button("🗑️ Clear All Indexed Data"):
        st.session_state.hybrid_service.child_chunks = []
        st.session_state.hybrid_service.parent_sections = {}
        st.session_state.hybrid_service.graph_store.clear()
        st.session_state.chat_history = []
        st.session_state.indexed_docs_meta = []
        st.rerun()


# ==================== MAIN TABS ====================
tab_chat, tab_graph, tab_inspector, tab_ragas = st.tabs([
    "🧠 Autonomous Research Chat",
    "🕸️ Knowledge Graph Explorer",
    "📑 Parent-Child Inspector",
    "📊 RAGAS Benchmarking"
])

# ----------------- TAB 1: CHAT -----------------
with tab_chat:
    st.markdown("### 🧠 NexusRAG Autonomous Research Interface")
    st.markdown("Autonomous query decomposition, concurrent multi-source retrieval, deterministic code execution, and real-time token streaming.")

    # Specialist Engine Selector Controls
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        selected_specialists = st.multiselect(
            "⚡ Active Research Specialist Engines (Run Simultaneously in Parallel)",
            options=["vector", "graph", "web", "code"],
            default=["vector", "graph"],
            format_func=lambda x: {
                "vector": "⚡ Vector Store (Qdrant Dense + BM25)",
                "graph": "🕸️ GraphRAG (Knowledge Graph Subgraphs)",
                "web": "🌐 Live Web Grounding (Tavily Search Engine)",
                "code": "🐍 Python Sandbox (Deterministic Calculations)"
            }[x],
            help="Select 2 or more specialist engines to trigger simultaneous parallel execution!"
        )
    with col_s2:
        if len(selected_specialists) > 1:
            st.success(f"🚀 **Parallel Mode:** {len(selected_specialists)} specialists run simultaneously!")
        else:
            st.info("Single specialist active.")

    # Render History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source_type" in msg:
                badge_class = "badge-qdrant" if "Qdrant" in msg["source_type"] else "badge-web"
                lat_str = f" | Latency: **{msg['latency']}s**" if "latency" in msg else ""
                st.markdown(f"<span class='{badge_class}'>{msg['source_type']}</span> &nbsp; Confidence: **{msg.get('confidence_score', 0)*100:.0f}%**{lat_str}", unsafe_allow_html=True)

            if msg.get("code_result"):
                with st.expander("🐍 Executed Python Sandbox Calculation"):
                    st.code(msg.get("code_snippet", ""), language="python")
                    st.info(f"**Output:**\n{msg['code_result']}")

            if msg.get("citations"):
                with st.expander(f"📚 Verified Sources & Citations ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(f"**[{c['citation_index']}] {c['source']} (Page {c['page']})**")
                        st.caption(f"{c['snippet']}...")

            if msg.get("node_trace"):
                with st.expander("🔍 Multi-Agent Execution Trace"):
                    for tr in msg["node_trace"]:
                        st.text(tr)

    # Chat Input
    query = st.chat_input("Ask a question, document summary, comparative query, or quantitative problem...")

    if query:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Agent execution with real-time status and token streaming
        with st.chat_message("assistant"):
            status_box = st.status("⚡ Dispatching NexusRAG Multi-Agent Orchestrator...", expanded=True)
            response_placeholder = st.empty()
            streamed_text = ""
            final_data = {}
            start_time = time.time()

            try:
                for event in st.session_state.orchestrator.run_stream(
                    query=query,
                    enabled_specialists=selected_specialists,
                    chat_history=st.session_state.chat_history
                ):
                    if event["type"] == "status":
                        status_box.write(event["message"])
                    elif event["type"] == "token":
                        streamed_text += event["content"]
                        response_placeholder.markdown(streamed_text + "▌")
                    elif event["type"] == "final_result":
                        final_data = event["data"]

                latency = round(time.time() - start_time, 2)
                num_specs = len(selected_specialists)
                spec_label = f"{num_specs} specialists simultaneously in parallel" if num_specs > 1 else "specialist"
                status_box.update(
                    label=f"✅ Research Completed in {latency}s ({spec_label})",
                    state="complete",
                    expanded=False
                )

                final_text = final_data.get("final_answer", streamed_text)
                response_placeholder.markdown(final_text)

                src_type = final_data.get("source_type", "NexusRAG Multi-Source")
                conf = final_data.get("confidence_score", 0.0)
                code_snippet = final_data.get("code_generated")
                code_out = final_data.get("code_execution_result")
                trace = final_data.get("node_trace", [])
                citations = final_data.get("citations", [])

                badge_class = "badge-qdrant" if "Qdrant" in src_type else "badge-web"
                st.markdown(f"<span class='{badge_class}'>{src_type}</span> &nbsp; Confidence: **{conf*100:.0f}%** | Latency: **{latency}s**", unsafe_allow_html=True)

                if code_out:
                    with st.expander("🐍 Python Code Sandbox Calculation"):
                        if code_snippet:
                            st.code(code_snippet, language="python")
                        st.info(f"**Execution Output:**\n{code_out}")

                if citations:
                    with st.expander(f"📚 Verified Sources & Citations ({len(citations)})"):
                        for c in citations:
                            st.markdown(f"**[{c['citation_index']}] {c['source']} (Page {c['page']})**")
                            st.caption(f"{c['snippet']}...")

                if trace:
                    with st.expander("🔍 Multi-Agent Execution Trace"):
                        for step in trace:
                            st.text(step)

                # Append to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "source_type": src_type,
                    "confidence_score": conf,
                    "latency": latency,
                    "code_snippet": code_snippet,
                    "code_result": code_out,
                    "node_trace": trace,
                    "citations": citations
                })

            except Exception as ex:
                status_box.update(label=f"❌ Error during execution: {ex}", state="error", expanded=True)
                st.error(f"Execution failed: {ex}")


# ----------------- TAB 2: KNOWLEDGE GRAPH -----------------
with tab_graph:
    st.markdown("### 🕸️ GraphRAG Knowledge Graph Explorer")
    st.markdown("Visualizes extracted entity-relation-entity triples and semantic connection networks.")

    g_nodes = st.session_state.hybrid_service.graph_store.graph.number_of_nodes()
    g_edges = st.session_state.hybrid_service.graph_store.graph.number_of_edges()

    if g_nodes > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Graph Entities", g_nodes)
        c2.metric("Total Relational Edges", g_edges)
        c3.metric("Graph Density", f"{g_edges / max(1, g_nodes):.2f}")

        vis_data = st.session_state.hybrid_service.graph_store.get_visualization_data(max_nodes=100)
        edges_df = pd.DataFrame([
            {"Subject (Entity)": e["from"], "Predicate (Relation)": e["label"], "Object (Target)": e["to"]}
            for e in vis_data["edges"]
        ])
        st.markdown("#### 🔗 Extracted Relational Triples")
        st.dataframe(edges_df, use_container_width=True)
    else:
        st.info("Upload and index documents to extract and inspect the Knowledge Graph.")


# ----------------- TAB 3: DOCUMENT INSPECTOR -----------------
with tab_inspector:
    st.markdown("### 📑 Parent-Child Document Hierarchy Inspector")
    st.markdown("Inspect how large parent context sections are linked to atomic child chunks for precision retrieval.")

    chunks = st.session_state.hybrid_service.child_chunks
    parents = st.session_state.hybrid_service.parent_sections

    if chunks:
        c1, c2 = st.columns(2)
        c1.metric("Total Atomic Child Chunks", len(chunks))
        c2.metric("Total Parent Sections", len(parents))

        selected_chunk_idx = st.selectbox(
            "Select Child Chunk to Inspect",
            options=range(len(chunks)),
            format_func=lambda i: f"Chunk {i+1}: {chunks[i]['content'][:60]}... (Page {chunks[i]['metadata'].get('page', 1)})"
        )

        selected_chunk = chunks[selected_chunk_idx]
        p_id = selected_chunk.get("parent_id")
        parent_obj = parents.get(p_id, {})

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🔍 Atomic Child Chunk (Vector/BM25 Index)")
            st.info(selected_chunk["content"])
            st.json(selected_chunk["metadata"])

        with col_r:
            st.markdown("#### 📖 Parent Context Block (Sent to LLM)")
            st.success(parent_obj.get("content", selected_chunk.get("parent_content", "No parent context")))
            if parent_obj.get("metadata", {}).get("is_table"):
                st.caption("📊 Contains structured table block")
    else:
        st.info("Upload documents to view the parent-child chunk hierarchy.")


# ----------------- TAB 4: RAGAS BENCHMARK -----------------
with tab_ragas:
    st.markdown("### 📊 Automated RAGAS Quality Benchmarking")
    st.markdown("Evaluates response Faithfulness, Context Precision, and the true Harmonic Mean ($F_1$) Score.")

    if st.button("⚡ Run Quantitative Benchmark"):
        with st.spinner("Benchmarking test queries against active index..."):
            test_queries = [
                "What are the core technical advantages of GraphRAG over pure vector search?",
                "How does parent-child hierarchical chunking prevent information loss?",
                "What is the function of the Citation Verification Critic?"
            ]

            results = []
            for t_q in test_queries:
                t0 = time.time()
                res = st.session_state.orchestrator.run(t_q)
                lat = round(time.time() - t0, 2)

                contexts = [c.get("content", "") for c in res.get("graded_documents", [])] or ["General domain knowledge context"]
                eval_out = st.session_state.evaluator.evaluate_response(t_q, res.get("final_answer", ""), contexts)

                results.append({
                    "Query": t_q,
                    "Faithfulness": eval_out["faithfulness"],
                    "Context Precision": eval_out["context_precision"],
                    "Harmonic RAGAS": eval_out["ragas_score"],
                    "Latency (s)": lat
                })

            df = pd.DataFrame(results)
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Faithfulness", f"{df['Faithfulness'].mean()*100:.1f}%")
            col2.metric("Average Precision", f"{df['Context Precision'].mean()*100:.1f}%")
            col3.metric("Average RAGAS F1", f"{df['Harmonic RAGAS'].mean()*100:.1f}%")

            st.dataframe(df, use_container_width=True)
