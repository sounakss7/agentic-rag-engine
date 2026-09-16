import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import config, register_degraded_component, get_degraded_components, is_degraded_mode
from retriever import HybridRetriever

logger = logging.getLogger("CRAGNodes")
logger.setLevel(logging.INFO)

try:
    from google import genai
    from google.genai import types
except ImportError:
    register_degraded_component("Gemini GenAI (stub)")
    genai = None
    types = None


try:
    from tavily import TavilyClient
except ImportError:
    register_degraded_component("TavilyClient (stub)")
    class TavilyClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
        def search(self, **kwargs):
            return {"results": [{"title": "Web Search Result", "content": "Information retrieved from Tavily web search fallback.", "url": "https://tavily.com"}]}


try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    register_degraded_component("LangGraph (stub)")
    START = "START"
    END = "END"
    class StateGraph:
        def __init__(self, state_schema):
            self.nodes = {}
            self.edges = {}
            self.cond_edges = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, src, dst):
            self.edges[src] = dst
        def add_conditional_edges(self, src, router_func, mapping):
            self.cond_edges[src] = (router_func, mapping)
        def compile(self):
            return RunnableFallbackGraph(self)

    class RunnableFallbackGraph:
        def __init__(self, graph):
            self.graph = graph
        def invoke(self, initial_state):
            current = "hyde_generator"
            state = initial_state
            visited = set()
            while current != END and current in self.graph.nodes:
                if current in visited and len(visited) > 10:
                    break
                visited.add(current)
                node_fn = self.graph.nodes[current]
                updates = node_fn(state)
                state.update(updates)
                if current in self.graph.cond_edges:
                    router_fn, mapping = self.graph.cond_edges[current]
                    next_node_name = router_fn(state)
                    current = mapping.get(next_node_name, END)
                elif current in self.graph.edges:
                    current = self.graph.edges[current]
                else:
                    break
            return state



# Pydantic Schemas for Structured Output
class GradeDocument(BaseModel):
    chunk_index: int = Field(default=1, description="1-based index of the document chunk.")
    is_relevant: bool = Field(description="True if document contains information relevant to the user query.")
    score: float = Field(description="Relevance confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Short rationale for the grade.")


class GradeDocumentsList(BaseModel):
    grades: List[GradeDocument] = Field(description="List of relevance grades for each document chunk.")


# LangGraph State Schema
class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, Any]]
    hyde_doc: str
    documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    generation: str
    confidence_score: float
    source_type: str
    fallback_required: bool
    node_trace: List[str]


class CRAGGraph:
    """
    Corrective RAG (CRAG) Orchestration Graph built with LangGraph and Gemini 2.5 Flash.
    Flow:
    1. HyDE Generator -> Generate hypothetical answer context vector.
    2. Hybrid Retrieval -> Search Qdrant Dense (with HyDE) + BM25 Sparse (clean keywords) & FlashRank Rerank.
    3. Context Grader -> Single-prompt batched relevance evaluation of chunks via LLM Judge.
    4. Conditional Router:
       - Valid context -> Answer Generation.
       - Invalid/low relevance context -> Fallback Search (Tavily Query Rewrite & Search) -> Answer Generation.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.graph = self._build_graph()

    def get_client(self) -> Optional[genai.Client]:
        """Dynamically retrieves GenAI client using active GEMINI_API_KEY."""
        if genai is not None and config.GEMINI_API_KEY:
            try:
                return genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"GenAI client init error: {e}")
        return None

    def _generate_with_retry(
        self,
        prompt: str,
        schema: Any = None,
        mime_type: Optional[str] = None,
        max_retries: int = 2
    ) -> Optional[str]:
        """Calls Gemini API with automatic exponential backoff retry and model cascade for HTTP 429."""
        client = self.get_client()
        if not client:
            return None

        cfg = None
        if schema is not None and types is not None:
            cfg = types.GenerateContentConfig(
                response_mime_type=mime_type or "application/json",
                response_schema=schema,
            )

        models_to_try = [config.LLM_MODEL, "gemini-flash-lite-latest", "gemini-flash-latest"]
        import time

        for attempt in range(max_retries + 1):
            for model_name in models_to_try:
                try:
                    kwargs = {"model": model_name, "contents": prompt}
                    if cfg is not None:
                        kwargs["config"] = cfg
                    res = client.models.generate_content(**kwargs)
                    if res and res.text:
                        return res.text.strip()
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Gemini rate limit (429) on {model_name}. Trying model cascade / backoff {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Gemini generation error on {model_name}: {e}")
                        break
        return None

    # ==================== NODE DEFINITIONS ====================

    def hyde_generator_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: HyDE Generator (Hypothetical Document Embeddings)"""
        query = state["query"]
        chat_history = state.get("chat_history", [])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 1: HyDE Generator] Generating hypothetical answer document...")

        # Build conversational context if available
        history_context = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            turns_str = "\n".join(f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in recent_turns)
            history_context = f"Recent Conversation:\n{turns_str}\n\n"

        hyde_doc = query
        prompt = (
            f"You are an expert technical researcher. Please write a concise hypothetical paragraph "
            f"or ideal answer that directly answers the user's question.\n"
            f"{history_context}"
            f"Question: {query}\n\n"
            f"Hypothetical Answer:"
        )
        generated = self._generate_with_retry(prompt)
        if generated:
            hyde_doc = generated

        return {
            "hyde_doc": hyde_doc,
            "node_trace": trace
        }

    def hybrid_retrieval_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Hybrid Retrieval Node (Qdrant Dense + BM25 Sparse + FlashRank Rerank)"""
        query = state["query"]
        hyde_doc = state.get("hyde_doc", query)
        trace = list(state.get("node_trace", []))
        trace.append("[Node 2: Hybrid Retrieval] Searching Qdrant Dense + BM25 with RRF & FlashRank...")

        # Dense query incorporates HyDE semantic context; Sparse & FlashRank strictly use clean user query
        dense_query = f"{query} {hyde_doc[:200]}"
        retrieved_docs = self.retriever.search(query, top_n=config.TOP_N_RERANK, dense_query=dense_query)

        trace.append(f"   [+] Retrieved {len(retrieved_docs)} candidate chunks from Vector Base.")
        return {
            "documents": retrieved_docs,
            "node_trace": trace
        }

    def context_grader_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Context Grading Node (Evaluates document relevance using Gemini LLM Judge in a single batch)"""
        query = state["query"]
        documents = state.get("documents", [])
        trace = list(state.get("node_trace", []))
        trace.append("[Node 3: Context Grader] Grading document relevance against query...")

        graded_docs = []
        fallback_required = False

        if not documents:
            trace.append("   [!] No documents retrieved. Triggering fallback route.")
            return {
                "graded_documents": [],
                "fallback_required": True,
                "node_trace": trace
            }

        client = self.get_client()
        graded_success = False

        # Attempt batched LLM grading (1 API call for all chunks)
        if client and types is not None:
            chunks_formatted = "\n\n".join([
                f"--- [Document Chunk {idx+1}] ---\n{doc['content']}"
                for idx, doc in enumerate(documents)
            ])
            prompt = (
                f"You are a strict relevance evaluator. Determine which of the following document chunks "
                f"contain information relevant to answering the user query.\n\n"
                f"User Query: {query}\n\n"
                f"{chunks_formatted}\n\n"
                f"For each chunk, provide chunk_index (1 to {len(documents)}), is_relevant (true/false), score (0.0 to 1.0), and reasoning."
            )
            try:
                raw_text = self._generate_with_retry(
                    prompt=prompt,
                    schema=GradeDocumentsList,
                    mime_type="application/json",
                )
                if raw_text:
                    clean_text = raw_text.strip().replace("```json", "").replace("```", "")
                    parsed = json.loads(clean_text)
                    grades_list = parsed.get("grades", [])

                    grades_by_idx = {g.get("chunk_index", i+1): g for i, g in enumerate(grades_list)}

                    for idx, doc in enumerate(documents):
                        grade_info = grades_by_idx.get(idx + 1, {})
                        is_rel = grade_info.get("is_relevant", True)
                        score = float(grade_info.get("score", 0.8))
                        reasoning = grade_info.get("reasoning", "Batched evaluation")

                        graded_item = doc.copy()
                        graded_item["is_relevant"] = is_rel
                        graded_item["relevance_score"] = score
                        graded_item["grading_reasoning"] = reasoning

                        if is_rel and score >= config.RELEVANCE_THRESHOLD:
                            graded_docs.append(graded_item)

                    graded_success = True
            except Exception as e:
                logger.warning(f"Batch grading LLM error: {e}. Falling back to individual evaluation.")

        # Fallback to keyword overlap heuristic if LLM grading was unavailable or failed
        if not graded_success:
            stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "on", "at", "by", "with", "from"}
            query_words = set(w for w in query.lower().split() if w not in stop_words)
            if not query_words:
                query_words = set(query.lower().split())

            for doc in documents:
                doc_words = set(doc["content"].lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = min(1.0, overlap / max(1, len(query_words)))
                is_rel = overlap > 0 and score >= 0.3

                graded_item = doc.copy()
                graded_item["is_relevant"] = is_rel
                graded_item["relevance_score"] = score
                graded_item["grading_reasoning"] = f"Keyword overlap ({overlap} matches)"

                if is_rel:
                    graded_docs.append(graded_item)

        if len(graded_docs) == 0:
            fallback_required = True
            trace.append("   [!] Context grading rejected all chunks. Routing to Web Search Fallback.")
        else:
            trace.append(f"   [+] Context grading passed {len(graded_docs)}/{len(documents)} relevant chunks.")

        return {
            "graded_documents": graded_docs,
            "fallback_required": fallback_required,
            "node_trace": trace
        }

    def fallback_search_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Fallback Node (Tavily Web Search + Query Rewriter)"""
        query = state["query"]
        trace = list(state.get("node_trace", []))
        trace.append("[Node 4: Corrective Fallback] Rewriting query and performing Web Search...")

        rewritten_query = query
        client = self.get_client()
        if client:
            prompt = (
                f"You are a search query optimizer. Rewrite the following user question into a clean, "
                f"effective search query for web retrieval:\nQuestion: {query}\n\nSearch Query:"
            )
            try:
                res_text = self._generate_with_retry(prompt)
                if res_text:
                    rewritten_query = res_text.strip()
            except Exception:
                pass

        web_chunks = []

        if config.TAVILY_API_KEY:
            try:
                tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
                search_resp = tavily.search(query=rewritten_query, search_depth="basic", max_results=3)
                results = search_resp.get("results", [])
                for idx, r in enumerate(results):
                    web_chunks.append({
                        "chunk_id": f"tavily_web_{idx}",
                        "content": f"Title: {r.get('title', '')}\nSnippet: {r.get('content', '')}",
                        "metadata": {
                            "source": f"Tavily Web: {r.get('url', 'web')}",
                            "page": 1,
                            "ocr_used": False,
                            "url": r.get('url', '')
                        },
                        "is_relevant": True,
                        "relevance_score": 0.9
                    })
                trace.append(f"   [+] Retrieved {len(web_chunks)} live web results via Tavily.")
            except Exception as e:
                logger.warning(f"Tavily API search error: {e}")
                trace.append(f"   [!] Tavily search error: {e}")

        if web_chunks:
            return {
                "graded_documents": web_chunks,
                "source_type": "Corrected via Tavily Web Search",
                "node_trace": trace
            }
        else:
            trace.append("   [!] External web search unavailable or returned no results.")
            return {
                "graded_documents": [],
                "source_type": "Knowledge Base (No external web fallback available)",
                "node_trace": trace
            }

    def answer_generator_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Final Answer Generation Node with citations and confidence score"""
        query = state["query"]
        chat_history = state.get("chat_history", [])
        documents = state.get("graded_documents", [])
        fallback_required = state.get("fallback_required", False)
        source_type = state.get("source_type", "Retrieved from Qdrant Vector Store" if not fallback_required else "Corrected via Tavily Web Search")
        trace = list(state.get("node_trace", []))
        trace.append("[Node 5: Answer Generator] Synthesizing final response with citations...")

        context_str = ""
        citations = []
        total_score = 0.0

        for idx, doc in enumerate(documents):
            source_name = doc.get("metadata", {}).get("source", f"Doc-{idx+1}")
            page_num = doc.get("metadata", {}).get("page", 1)
            context_str += f"\n--- [Source {idx+1}: {source_name} (Page {page_num})] ---\n{doc['content']}\n"
            citations.append(f"[{idx+1}] {source_name} (Page {page_num})")
            total_score += doc.get("relevance_score", 0.8)

        confidence_score = round(total_score / max(1, len(documents)), 2) if documents else 0.0

        # Build conversation context
        history_context = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            turns_str = "\n".join(f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in recent_turns)
            history_context = f"Prior Conversation:\n{turns_str}\n\n"

        client = self.get_client()
        if client:
            if documents:
                prompt = (
                    f"You are an enterprise AI assistant trained on Advanced Corrective RAG.\n"
                    f"Answer the user's question thoroughly using the provided context chunks.\n"
                    f"Include clear numerical source citations [1], [2], etc. matching the source documents.\n"
                    f"If the context is insufficient, state what is known and clarify limitations.\n\n"
                    f"{history_context}"
                    f"User Question: {query}\n\n"
                    f"Context Chunks:\n{context_str}\n\n"
                    f"Detailed Answer:"
                )
            else:
                prompt = (
                    f"You are an enterprise AI assistant trained on Advanced Corrective RAG.\n"
                    f"The user asked: '{query}'.\n"
                    f"Neither the uploaded document vector store nor external web fallback yielded relevant information.\n"
                    f"Please politely inform the user that no relevant information was found in the indexed documents, "
                    f"and clarify what document or details would be needed to answer."
                )
            try:
                res_text = self._generate_with_retry(prompt)
                answer = res_text.strip() if res_text else (f"Synthesized Answer based on context: {context_str[:400]}..." if documents else "No relevant documents found.")
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
                answer = f"Synthesized Answer based on context: {context_str[:400]}..." if documents else "No relevant documents found."
        else:
            answer = f"Synthesized Answer based on retrieved context:\n{context_str[:600]}" if documents else "No relevant documents found in index."

        trace.append(f"   [+] Finished generation. Confidence: {confidence_score * 100:.0f}%. Source: {source_type}")

        return {
            "generation": answer,
            "confidence_score": confidence_score,
            "source_type": source_type,
            "node_trace": trace
        }

    # ==================== ROUTING LOGIC ====================

    @staticmethod
    def decide_to_generate(state: GraphState) -> str:
        """Conditional Router Edge decision."""
        if state.get("fallback_required", False):
            return "fallback_search"
        return "answer_generator"

    # ==================== GRAPH BUILDER ====================

    def _build_graph(self) -> Any:
        """Constructs and compiles the LangGraph StateGraph workflow."""
        workflow = StateGraph(GraphState)

        workflow.add_node("hyde_generator", self.hyde_generator_node)
        workflow.add_node("hybrid_retrieval", self.hybrid_retrieval_node)
        workflow.add_node("context_grader", self.context_grader_node)
        workflow.add_node("fallback_search", self.fallback_search_node)
        workflow.add_node("answer_generator", self.answer_generator_node)

        workflow.add_edge(START, "hyde_generator")
        workflow.add_edge("hyde_generator", "hybrid_retrieval")
        workflow.add_edge("hybrid_retrieval", "context_grader")

        workflow.add_conditional_edges(
            "context_grader",
            self.decide_to_generate,
            {
                "fallback_search": "fallback_search",
                "answer_generator": "answer_generator",
            }
        )

        workflow.add_edge("fallback_search", "answer_generator")
        workflow.add_edge("answer_generator", END)

        return workflow.compile()

    def run(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Executes the CRAG Graph workflow for a user query with optional chat history."""
        initial_state: GraphState = {
            "query": query,
            "chat_history": chat_history or [],
            "hyde_doc": "",
            "documents": [],
            "graded_documents": [],
            "generation": "",
            "confidence_score": 0.0,
            "source_type": "Retrieved from Qdrant Vector Store",
            "fallback_required": False,
            "node_trace": []
        }

        final_state = self.graph.invoke(initial_state)
        return final_state

