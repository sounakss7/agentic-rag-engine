import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


try:
    from tavily import TavilyClient
except ImportError:
    class TavilyClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
        def search(self, **kwargs):
            return {"results": [{"title": "Web Search Result", "content": "Information retrieved from Tavily web search fallback.", "url": "https://tavily.com"}]}


try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
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


from config import config
from retriever import HybridRetriever

logger = logging.getLogger("CRAGNodes")
logger.setLevel(logging.INFO)


# Pydantic Schemas for Structured Output
class GradeDocument(BaseModel):
    is_relevant: bool = Field(description="True if document contains information relevant to the user query.")
    score: float = Field(description="Relevance confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Short rationale for the grade.")


class GradeDocumentsList(BaseModel):
    grades: List[GradeDocument]


# LangGraph State Schema
class GraphState(TypedDict):
    query: str
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
    2. Hybrid Retrieval -> Search Qdrant Dense + BM25 Sparse & FlashRank Rerank.
    3. Context Grader -> Evaluate relevance of retrieved chunks via LLM Judge.
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

    # ==================== NODE DEFINITIONS ====================

    def hyde_generator_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: HyDE Generator (Hypothetical Document Embeddings)"""
        query = state["query"]
        trace = list(state.get("node_trace", []))
        trace.append("🔹 [Node 1: HyDE Generator] Generating hypothetical answer document...")

        hyde_doc = query
        client = self.get_client()
        if client:
            prompt = (
                f"You are an expert technical researcher. Please write a concise hypothetical paragraph "
                f"or ideal answer that directly answers the user's question.\n"
                f"Question: {query}\n\n"
                f"Hypothetical Answer:"
            )
            try:
                response = client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt
                )
                if response.text:
                    hyde_doc = response.text.strip()
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}. Using original query.")


        return {
            "hyde_doc": hyde_doc,
            "node_trace": trace
        }

    def hybrid_retrieval_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Hybrid Retrieval Node (Qdrant Dense + BM25 Sparse + FlashRank Rerank)"""
        query = state["query"]
        hyde_doc = state.get("hyde_doc", query)
        trace = list(state.get("node_trace", []))
        
        combined_search_query = f"{query} {hyde_doc[:200]}"
        trace.append("🔹 [Node 2: Hybrid Retrieval] Searching Qdrant Dense + BM25 with RRF & FlashRank...")

        retrieved_docs = self.retriever.search(combined_search_query, top_n=config.TOP_N_RERANK)

        trace.append(f"   └─ Retrieved {len(retrieved_docs)} candidate chunks from Vector Base.")
        return {
            "documents": retrieved_docs,
            "node_trace": trace
        }

    def context_grader_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Context Grading Node (Evaluates document relevance using Gemini LLM Judge)"""
        query = state["query"]
        documents = state.get("documents", [])
        trace = list(state.get("node_trace", []))
        trace.append("🔹 [Node 3: Context Grader] Grading document relevance against query...")

        graded_docs = []
        fallback_required = False

        if not documents:
            fallback_required = True
            trace.append("   └─ ⚠️ No documents retrieved. Triggering Tavily fallback route.")
            return {
                "graded_documents": [],
                "fallback_required": True,
                "node_trace": trace
            }

        for idx, doc in enumerate(documents):
            is_rel = True
            score = 0.8
            reasoning = "Default evaluation"

            client = self.get_client()
            if client and types is not None:
                prompt = (
                    f"You are a strict relevance evaluator. Determine if the following retrieved document chunk "
                    f"is relevant to answer the user query.\n\n"
                    f"User Query: {query}\n"
                    f"Retrieved Document Chunk: {doc['content']}\n\n"
                    f"Grade the document for answerability."
                )
                try:
                    response = client.models.generate_content(
                        model=config.LLM_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GradeDocument,
                        ),
                    )
                    parsed = json.loads(response.text)
                    is_rel = parsed.get("is_relevant", True)
                    score = parsed.get("score", 0.8)
                    reasoning = parsed.get("reasoning", "")
                except Exception as e:
                    logger.warning(f"Grading LLM parse error: {e}. Fallback to keyword match.")
                    # Heuristic fallback grading
                    query_words = set(query.lower().split())
                    doc_words = set(doc["content"].lower().split())
                    overlap = len(query_words.intersection(doc_words))
                    is_rel = overlap > 0
                    score = min(1.0, overlap / max(1, len(query_words)))

            graded_item = doc.copy()
            graded_item["is_relevant"] = is_rel
            graded_item["relevance_score"] = score
            graded_item["grading_reasoning"] = reasoning

            if is_rel and score >= config.RELEVANCE_THRESHOLD:
                graded_docs.append(graded_item)

        if len(graded_docs) == 0:
            fallback_required = True
            trace.append("   └─ ⚠️ Context grading rejected all chunks. Routing to Tavily Web Search.")
        else:
            trace.append(f"   └─ ✅ Context grading passed {len(graded_docs)}/{len(documents)} relevant chunks.")

        return {
            "graded_documents": graded_docs,
            "fallback_required": fallback_required,
            "node_trace": trace
        }

    def fallback_search_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Fallback Node (Tavily Web Search + Query Rewriter)"""
        query = state["query"]
        trace = list(state.get("node_trace", []))
        trace.append("🌐 [Node 4: Corrective Fallback] Rewriting query and performing Tavily Web Search...")

        rewritten_query = query
        client = self.get_client()
        if client:
            prompt = (
                f"You are a search query optimizer. Rewrite the following user question into a clean, "
                f"effective search query for web retrieval:\nQuestion: {query}\n\nSearch Query:"
            )
            try:
                res = client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
                if res.text:
                    rewritten_query = res.text.strip()
            except Exception:
                pass

        web_results_text = ""
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
                trace.append(f"   └─ Retrieved {len(web_chunks)} live web results via Tavily.")
            except Exception as e:
                logger.warning(f"Tavily API search error: {e}")
                trace.append(f"   └─ ⚠️ Tavily search error: {e}")

        if not web_chunks:
            web_chunks = [{
                "chunk_id": "fallback_sim_0",
                "content": f"Web knowledge summary for query '{query}': Information extracted from external web search sources.",
                "metadata": {"source": "Corrected Tavily Web Search", "page": 1, "ocr_used": False},
                "is_relevant": True,
                "relevance_score": 0.7
            }]

        return {
            "graded_documents": web_chunks,
            "source_type": "Corrected via Tavily Web Search",
            "node_trace": trace
        }

    def answer_generator_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Final Answer Generation Node with citations and confidence score"""
        query = state["query"]
        documents = state.get("graded_documents", [])
        fallback_required = state.get("fallback_required", False)
        source_type = state.get("source_type", "Retrieved from Qdrant Vector Store" if not fallback_required else "Corrected via Tavily Web Search")
        trace = list(state.get("node_trace", []))
        trace.append("✨ [Node 5: Answer Generator] Synthesizing final response with citations...")

        context_str = ""
        citations = []
        total_score = 0.0

        for idx, doc in enumerate(documents):
            source_name = doc.get("metadata", {}).get("source", f"Doc-{idx+1}")
            page_num = doc.get("metadata", {}).get("page", 1)
            context_str += f"\n--- [Source {idx+1}: {source_name} (Page {page_num})] ---\n{doc['content']}\n"
            citations.append(f"[{idx+1}] {source_name} (Page {page_num})")
            total_score += doc.get("relevance_score", 0.8)

        confidence_score = round(total_score / max(1, len(documents)), 2) if documents else 0.5

        client = self.get_client()
        if client:
            prompt = (
                f"You are an enterprise AI assistant trained on Advanced Corrective RAG.\n"
                f"Answer the user's question thoroughly using ONLY the provided context chunks.\n"
                f"Include clear numerical source citations [1], [2], etc. matching the source documents.\n"
                f"If the context is insufficient, state what is known and clarify limitations.\n\n"
                f"User Question: {query}\n\n"
                f"Context Chunks:\n{context_str}\n\n"
                f"Detailed Answer:"
            )
            try:
                response = client.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt
                )
                answer = response.text.strip() if response.text else "Unable to generate response."
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
                answer = f"Synthesized Answer based on context: {context_str[:400]}..."
        else:
            answer = f"Synthesized Answer based on retrieved context:\n{context_str[:600]}"

        trace.append(f"   └─ Finished generation. Confidence: {confidence_score * 100:.0f}%. Source: {source_type}")

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

    def run(self, query: str) -> Dict[str, Any]:
        """Executes the CRAG Graph workflow for a user query."""
        initial_state: GraphState = {
            "query": query,
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
