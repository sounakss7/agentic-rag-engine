import time
import streamlit as st
import pandas as pd

# Internal Module Imports
from config import config
from ocr_parser import DocumentIngestor
from retriever import HybridRetriever
from graph_nodes import CRAGGraph
from eval_pipeline import RAGASEvaluator

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Enterprise Advanced CRAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING (CSS) ====================
CUSTOM_CSS = """
<style>
    /* Dark / Glassmorphic Enterprise Theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Source Badges */
    .badge-qdrant {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-tavily {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Confidence Badge */
    .confidence-badge {
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(165, 180, 252, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
    }

    /* Trace Log box */
    .trace-box {
        background-color: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        line-height: 1.6;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================== INITIALIZATION ====================
def initialize_session_state():
    """Initializes persistent application session state objects."""
    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever()

    if "crag_graph" not in st.session_state:
        st.session_state.crag_graph = CRAGGraph(st.session_state.retriever)

    if "ingestor" not in st.session_state:
        st.session_state.ingestor = DocumentIngestor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = RAGASEvaluator()

    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "eval_results_df" not in st.session_state:
        st.session_state.eval_results_df = None

    if "eval_summary" not in st.session_state:
        st.session_state.eval_summary = None


initialize_session_state()


# ==================== SIDEBAR UI ====================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/brain.png", width=64)
    st.title("CRAG Control Center")
    st.caption("Enterprise Advanced Corrective RAG System")
    st.divider()

    # --- Section 1: Key Status Indicators ---
    st.subheader("🔑 API Key Status")
    key_status = config.key_status()

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown(f"**Gemini:** {'✅ Active' if key_status['GEMINI_API_KEY'] else '⚠️ Missing'}")
        st.markdown(f"**Tavily:** {'✅ Active' if key_status['TAVILY_API_KEY'] else '⚠️ Missing'}")
    with col_k2:
        qdrant_mode = "Cloud ✅" if key_status['QDRANT_URL'] else "Local Memory 💾"
        st.markdown(f"**Qdrant:** {qdrant_mode}")
        st.markdown(f"**LangChain:** {'✅ Active' if key_status['LANGCHAIN_API_KEY'] else '⚪ Optional'}")

    with st.expander("⚙️ Configure / Override API Keys"):
        st.caption("If not set in Streamlit Secrets, enter your keys below:")
        input_gemini = st.text_input("Gemini API Key", type="password", key="ui_gemini_key", value=st.session_state.get("GEMINI_API_KEY", ""))
        input_tavily = st.text_input("Tavily API Key", type="password", key="ui_tavily_key", value=st.session_state.get("TAVILY_API_KEY", ""))
        input_qdrant_url = st.text_input("Qdrant URL (Optional)", key="ui_qdrant_url", value=st.session_state.get("QDRANT_URL", ""))
        input_qdrant_key = st.text_input("Qdrant API Key (Optional)", type="password", key="ui_qdrant_key", value=st.session_state.get("QDRANT_API_KEY", ""))
        
        if st.button("Save & Refresh Keys", use_container_width=True):
            if input_gemini.strip(): st.session_state["GEMINI_API_KEY"] = input_gemini.strip()
            if input_tavily.strip(): st.session_state["TAVILY_API_KEY"] = input_tavily.strip()
            if input_qdrant_url.strip(): st.session_state["QDRANT_URL"] = input_qdrant_url.strip()
            if input_qdrant_key.strip(): st.session_state["QDRANT_API_KEY"] = input_qdrant_key.strip()
            
            # Re-initialize graph and evaluator with updated keys
            st.session_state.crag_graph = CRAGGraph(st.session_state.retriever)
            st.session_state.evaluator = RAGASEvaluator()
            st.success("API keys updated!")
            st.rerun()

    st.divider()


    # --- Section 2: Document Ingestion & OCR ---
    st.subheader("📁 Document Ingestion & OCR")
    uploaded_files = st.file_uploader(
        "Upload files for indexing",
        type=["pdf", "png", "jpg", "jpeg", "txt", "md"],
        accept_multiple_files=True,
        help="Supports native PDF, scanned PDF OCR, images, and text files."
    )

    if st.button("🚀 Build / Refresh Vector Index", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document to index.")
        else:
            all_chunks = []
            progress_bar = st.progress(0, text="Starting document parsing & OCR pipeline...")
            
            for idx, uf in enumerate(uploaded_files):
                file_bytes = uf.read()
                filename = uf.name
                progress_bar.progress(
                    int(((idx + 0.5) / len(uploaded_files)) * 100),
                    text=f"Parsing & OCR: {filename}..."
                )
                chunks, summary = st.session_state.ingestor.process_file(file_bytes, filename)
                all_chunks.extend(chunks)
                st.toast(f"Parsed {filename}: {summary['total_chunks']} chunks ({summary['ocr_pages_count']} OCR pages)", icon="📄")

            progress_bar.progress(90, text="Indexing chunks into Qdrant & BM25 Sparse Store...")
            st.session_state.retriever.build_index(all_chunks)
            st.session_state.indexed_chunks = all_chunks
            
            # Refresh CRAG Graph reference
            st.session_state.crag_graph = CRAGGraph(st.session_state.retriever)

            progress_bar.progress(100, text="Vector index build complete!")
            time.sleep(0.5)
            progress_bar.empty()
            st.success(f"Successfully indexed {len(all_chunks)} total chunks into Hybrid Engine!")

    st.divider()

    # --- Section 3: Collection Statistics & Controls ---
    st.subheader("📊 Database Stats")
    chunk_count = len(st.session_state.indexed_chunks)
    st.metric("Total Indexed Chunks", chunk_count)
    st.metric("Vector Dimensions", config.EMBEDDING_DIM)

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        st.session_state.retriever.clear()
        st.session_state.indexed_chunks = []
        st.session_state.chat_history = []
        st.success("Database cleared.")
        st.rerun()


# ==================== MAIN CONTENT TABS ====================
st.title("⚡ Enterprise Advanced Corrective RAG (CRAG) Engine")
st.caption("Multi-node LangGraph Architecture with Qdrant Hybrid Search, FlashRank Reranking & Tavily Fallback Search")

tab1, tab2, tab3 = st.tabs([
    "💬 Interactive Chat Engine",
    "🔍 Document Inspector",
    "📈 RAGAS Evaluation Dashboard"
])


# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE CHAT ENGINE
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive CRAG Engine Chat")
    
    if not st.session_state.indexed_chunks:
        st.info("💡 **Getting Started**: Upload documents in the sidebar and click **Build Vector Index** to query your knowledge base. Queries outside your uploaded documents will trigger Tavily Fallback Web Search automatically.")

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # Render Badge
                badge_class = "badge-tavily" if "Tavily" in msg.get("source_type", "") else "badge-qdrant"
                st.markdown(f"""
                <span class="{badge_class}">{msg.get("source_type", "Retrieved Vector Base")}</span>
                <span class="confidence-badge">Confidence: {int(msg.get("confidence_score", 0.8) * 100)}%</span>
                """, unsafe_allow_html=True)
                
                st.markdown(msg["content"])

                # Expandable Node Trace
                if msg.get("node_trace"):
                    with st.expander("🔄 View LangGraph CRAG Execution Trace"):
                        trace_html = "<br>".join(msg["node_trace"])
                        st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)

                # Source Citations Popover
                if msg.get("documents"):
                    with st.popover("📚 View Source Citations & Chunks"):
                        for idx, doc in enumerate(msg["documents"]):
                            st.markdown(f"**[{idx+1}] Source:** `{doc.get('metadata', {}).get('source', 'Web')}` | **Page:** `{doc.get('metadata', {}).get('page', 1)}`")
                            st.caption(f"Relevance Score: {doc.get('relevance_score', 0.8):.2f}")
                            st.text_area(f"Chunk Text [{idx+1}]", doc["content"], height=110, key=f"hist_doc_{msg['id']}_{idx}")
            else:
                st.markdown(msg["content"])

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your documents or external domain...")

    if user_query:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_query, "id": len(st.session_state.chat_history)})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant processing
        with st.chat_message("assistant"):
            with st.spinner("Orchestrating CRAG LangGraph Nodes (HyDE ➔ Hybrid Retrieval ➔ Context Grading ➔ Synthesis)..."):
                start_t = time.time()
                graph_output = st.session_state.crag_graph.run(user_query)
                elapsed = round(time.time() - start_t, 2)

            ans_text = graph_output.get("generation", "No response generated.")
            source_type = graph_output.get("source_type", "Retrieved from Qdrant Vector Store")
            conf_score = graph_output.get("confidence_score", 0.85)
            node_trace = graph_output.get("node_trace", [])
            graded_docs = graph_output.get("graded_documents", [])

            # Render Badge
            badge_class = "badge-tavily" if "Tavily" in source_type else "badge-qdrant"
            st.markdown(f"""
            <span class="{badge_class}">{source_type}</span>
            <span class="confidence-badge">Confidence: {int(conf_score * 100)}%</span>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 10px;">(Latency: {elapsed}s)</span>
            """, unsafe_allow_html=True)

            st.markdown(ans_text)

            # Node Execution Trace Expander
            with st.expander("🔄 View LangGraph CRAG Execution Trace"):
                trace_html = "<br>".join(node_trace)
                st.markdown(f'<div class="trace-box">{trace_html}</div>', unsafe_allow_html=True)

            # Citations Popover
            if graded_docs:
                with st.popover("📚 View Source Citations & Chunks"):
                    for idx, doc in enumerate(graded_docs):
                        st.markdown(f"**[{idx+1}] Source:** `{doc.get('metadata', {}).get('source', 'Web')}` | **Page:** `{doc.get('metadata', {}).get('page', 1)}`")
                        st.caption(f"Relevance Score: {doc.get('relevance_score', 0.8):.2f}")
                        st.text_area(f"Chunk Text [{idx+1}]", doc["content"], height=110, key=f"new_doc_{len(st.session_state.chat_history)}_{idx}")

            # Append to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ans_text,
                "source_type": source_type,
                "confidence_score": conf_score,
                "node_trace": node_trace,
                "documents": graded_docs,
                "id": len(st.session_state.chat_history)
            })


# -----------------------------------------------------------------------------
# TAB 2: DOCUMENT INSPECTOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Vector Database & Document Inspection")
    chunks = st.session_state.indexed_chunks

    if not chunks:
        st.info("No documents indexed in the vector store yet. Use the sidebar to upload files.")
    else:
        st.success(f"Currently inspecting {len(chunks)} chunked records.")
        
        # Filtering options
        sources = list(set(c["metadata"]["source"] for c in chunks))
        selected_source = st.selectbox("Filter by Source File", options=["All Sources"] + sources)

        filtered_chunks = chunks if selected_source == "All Sources" else [c for c in chunks if c["metadata"]["source"] == selected_source]

        for i, chk in enumerate(filtered_chunks[:30]):
            with st.expander(f"Chunk #{i+1} | Source: {chk['metadata']['source']} (Page {chk['metadata']['page']}) | ID: {chk['chunk_id']}"):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown("**Chunk Text Content:**")
                    st.write(chk["content"])
                with col_b:
                    st.markdown("**Metadata:**")
                    st.json(chk["metadata"])


# -----------------------------------------------------------------------------
# TAB 3: RAGAS EVALUATION DASHBOARD
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Automated RAGAS Evaluation Dashboard")
    st.write("Measures **Faithfulness** (hallucination check) and **Context Precision** (signal-to-noise ratio) across test benchmark queries.")

    # Benchmark Test Suite controls
    sample_queries = [
        {"query": "What are the core specifications and architecture described in the document?"},
        {"query": "What is the primary methodology or algorithm used?"},
        {"query": "Who is the president of Mars?"}  # Out of domain query to trigger fallback
    ]

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🧪 Run RAGAS Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running test queries through CRAG Graph and computing LLM Judge Metrics..."):
                df_results, summary = st.session_state.evaluator.run_benchmark_suite(
                    sample_queries,
                    st.session_state.crag_graph
                )
                st.session_state.eval_results_df = df_results
                st.session_state.eval_summary = summary
                st.success("Evaluation complete!")

    if st.session_state.eval_summary:
        summary = st.session_state.eval_summary
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Faithfulness Score</div>
                <div class="metric-value">{summary['avg_faithfulness'] * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Context Precision</div>
                <div class="metric-value">{summary['avg_precision'] * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overall RAGAS Score</div>
                <div class="metric-value">{summary['avg_ragas_score'] * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Latency</div>
                <div class="metric-value">{summary['avg_latency_s']}s</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Detailed Evaluation Results")
        st.dataframe(
            st.session_state.eval_results_df,
            use_container_width=True,
            column_config={
                "Faithfulness": st.column_config.ProgressColumn("Faithfulness", format="%.2f", min_value=0, max_value=1),
                "Context Precision": st.column_config.ProgressColumn("Context Precision", format="%.2f", min_value=0, max_value=1),
                "RAGAS Score": st.column_config.ProgressColumn("RAGAS Score", format="%.2f", min_value=0, max_value=1)
            }
        )
