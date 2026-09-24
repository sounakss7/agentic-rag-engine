"""
NexusRAG Knowledge Graph (GraphRAG) Store.
Extracts entity-relation-entity triples from parent sections and performs subgraph traversal for relational queries.
Provides interactive graph visualization data.
"""

import re
import json
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GraphTriple(BaseModel):
    subject: str = Field(description="Subject entity, e.g. Apple Inc, Revenue, Q3 2024")
    predicate: str = Field(description="Relationship or verb, e.g. REPORTED_REVENUE, INCREASED_BY, LOCATED_IN")
    object: str = Field(description="Object entity or value, e.g. $89.5 Billion, 14%, California")


class ExtractedTriplesList(BaseModel):
    triples: List[GraphTriple] = Field(default_factory=list)


class GraphStoreService:
    """
    Knowledge Graph (GraphRAG) management engine.
    Constructs an in-memory MultiDiGraph of entities, attributes, and relationships.
    Enables multi-hop associative queries that vector search misses.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._genai_client = None

    def _get_genai_client(self):
        if self._genai_client is None and genai is not None and settings.active_gemini_key:
            self._genai_client = genai.Client(api_key=settings.active_gemini_key)
        return self._genai_client

    def clear(self):
        """Resets the knowledge graph."""
        self.graph.clear()

    def build_graph_from_sections(self, parent_sections: Dict[str, Dict[str, Any]]) -> int:
        """
        Extracts entity-relation triples from parent sections and indexes them into the graph.
        """
        total_triples = 0
        client = self._get_genai_client()

        for parent_id, sec_data in parent_sections.items():
            content = sec_data.get("content", "")
            source = sec_data.get("metadata", {}).get("source", "doc")
            page = sec_data.get("metadata", {}).get("page", 1)

            if not content or len(content.strip()) < 30:
                continue

            extracted_triples = []

            # 1. LLM-Assisted Triple Extraction
            if client and types is not None:
                try:
                    prompt = (
                        "Extract the key factual relationships, financial metrics, and entities from this text "
                        "into clean (subject, predicate, object) triples.\n\n"
                        f"Text:\n{content[:1500]}\n\n"
                        "Extract up to 6 high-value triples."
                    )
                    res = client.models.generate_content(
                        model=settings.PRIMARY_LLM,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ExtractedTriplesList
                        )
                    )
                    if res and res.text:
                        parsed = json.loads(res.text.strip().replace("```json", "").replace("```", ""))
                        raw_list = parsed.get("triples", [])
                        for item in raw_list:
                            s = str(item.get("subject", "")).strip()
                            p = str(item.get("predicate", "")).strip().upper().replace(" ", "_")
                            o = str(item.get("object", "")).strip()
                            if s and p and o and len(s) < 100 and len(o) < 120:
                                extracted_triples.append((s, p, o))
                except Exception as e:
                    logger.debug(f"LLM triple extraction notice on {parent_id}: {e}")

            # 2. Rule-Based Fallback Extraction (Dates, percentages, currencies, colon-attributes)
            if not extracted_triples:
                extracted_triples = self._extract_heuristic_triples(content, source)

            # Insert into graph
            for s, p, o in extracted_triples:
                self.graph.add_node(s, label=s, type="entity")
                self.graph.add_node(o, label=o, type="value_or_entity")
                self.graph.add_edge(s, o, relation=p, source=source, page=page, parent_id=parent_id)
                total_triples += 1

        logger.info(f"Constructed Knowledge Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return total_triples

    def _extract_heuristic_triples(self, text: str, source: str) -> List[Tuple[str, str, str]]:
        """Fast regex extraction for structured attributes, metrics, and definitions."""
        triples = []
        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            # Attribute pattern: "Key: Value" or "Metric - Value"
            if ":" in line and not line.startswith("http"):
                parts = line.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if 2 < len(k) < 50 and 1 < len(v) < 100:
                    triples.append((k, "HAS_VALUE", v))

            # Financial / Numerical pattern: "increased by 25%", "total revenue of $10M"
            match = re.search(r"([A-Za-z\s]{3,30})\s+(increased by|decreased by|reached|totaled|reported)\s+([\$€£]?\d+[\d\.,]*%?|\w+)", line, re.IGNORECASE)
            if match:
                s, p, o = match.groups()
                triples.append((s.strip(), p.strip().upper().replace(" ", "_"), o.strip()))

        return triples[:5]

    def search_subgraph(self, query: str, top_k: int = settings.TOP_K_GRAPH) -> List[Dict[str, Any]]:
        """
        Discovers relevant subgraphs by entity matching and traverses 1-hop and 2-hop edges.
        Returns formatted relationship facts with source metadata.
        """
        if self.graph.number_of_edges() == 0:
            return []

        query_terms = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", query.lower()))
        matched_nodes = []

        # Find matching seed nodes
        for node in self.graph.nodes():
            node_str = str(node).lower()
            if any(term in node_str for term in query_terms):
                matched_nodes.append(node)

        if not matched_nodes:
            # Fallback: check edge relation attributes
            for u, v, data in self.graph.edges(data=True):
                rel = str(data.get("relation", "")).lower()
                if any(term in rel for term in query_terms):
                    matched_nodes.extend([u, v])

        matched_nodes = list(set(matched_nodes))[:8]
        results = []

        # Traverse 1-hop neighborhood for matched nodes
        seen_edges = set()
        for node in matched_nodes:
            # Outgoing edges
            for _, target, data in self.graph.out_edges(node, data=True):
                edge_id = (node, target, data.get("relation"))
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    rel = data.get("relation", "RELATED_TO")
                    fact_str = f"({node}) --[{rel}]--> ({target})"
                    results.append({
                        "chunk_id": f"graph_edge_{len(results)}",
                        "content": f"[Knowledge Graph Relation] {fact_str}",
                        "parent_id": data.get("parent_id", ""),
                        "metadata": {
                            "source": f"Knowledge Graph: {data.get('source', 'graph')}",
                            "page": data.get("page", 1),
                            "subject": str(node),
                            "relation": rel,
                            "object": str(target),
                            "is_graph": True
                        },
                        "score": 0.88,
                        "search_type": "knowledge_graph"
                    })

            # Incoming edges
            for source, _, data in self.graph.in_edges(node, data=True):
                edge_id = (source, node, data.get("relation"))
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    rel = data.get("relation", "RELATED_TO")
                    fact_str = f"({source}) --[{rel}]--> ({node})"
                    results.append({
                        "chunk_id": f"graph_edge_{len(results)}",
                        "content": f"[Knowledge Graph Relation] {fact_str}",
                        "parent_id": data.get("parent_id", ""),
                        "metadata": {
                            "source": f"Knowledge Graph: {data.get('source', 'graph')}",
                            "page": data.get("page", 1),
                            "subject": str(source),
                            "relation": rel,
                            "object": str(node),
                            "is_graph": True
                        },
                        "score": 0.85,
                        "search_type": "knowledge_graph"
                    })

        return results[:top_k]

    def get_visualization_data(self, max_nodes: int = 60) -> Dict[str, Any]:
        """
        Exports nodes and edges in D3 / Cytoscape JSON format for frontend rendering.
        """
        nodes = []
        edges = []

        subgraph_nodes = list(self.graph.nodes())[:max_nodes]
        sub = self.graph.subgraph(subgraph_nodes)

        for n in sub.nodes(data=True):
            node_id = str(n[0])
            nodes.append({
                "id": node_id,
                "label": node_id[:25],
                "title": node_id,
                "group": n[1].get("type", "entity")
            })

        for u, v, data in sub.edges(data=True):
            edges.append({
                "from": str(u),
                "to": str(v),
                "label": str(data.get("relation", "RELATED_TO"))[:20],
                "title": f"{u} -> {data.get('relation')} -> {v}"
            })

        return {"nodes": nodes, "edges": edges, "total_nodes": self.graph.number_of_nodes(), "total_edges": self.graph.number_of_edges()}
