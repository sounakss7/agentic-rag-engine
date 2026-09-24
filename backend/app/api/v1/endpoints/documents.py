"""
Document ingestion, layout-aware parsing, and indexing endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.app.services.ingestion.parser import LayoutAwareParser
from backend.app.services.ingestion.chunker import HierarchicalChunker
from backend.app.services.retrieval.hybrid import HybridRetrievalService

router = APIRouter()

# Shared Global Service Singletons
parser = LayoutAwareParser()
chunker = HierarchicalChunker()
hybrid_service = HybridRetrievalService()


class IngestionSummaryResponse(BaseModel):
    status: str
    filename: str
    pages: int
    child_chunks: int
    parent_sections: int
    graph_triples: int


@router.post("/upload", response_model=IngestionSummaryResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads and indexes a document through layout-aware parent-child chunking and GraphRAG."""
    filename = file.filename or "uploaded_document"
    try:
        content = await file.read()
        pages_data = parser.parse_document(content, filename)
        child_chunks, parent_map = chunker.process_document(pages_data, filename)

        res = hybrid_service.build_index(child_chunks, parent_map)

        return IngestionSummaryResponse(
            status="success",
            filename=filename,
            pages=len(pages_data),
            child_chunks=len(child_chunks),
            parent_sections=len(parent_map),
            graph_triples=res.get("graph_triples", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get("/documents")
def list_documents():
    """Lists indexed document chunks and parent sections."""
    return {
        "total_child_chunks": len(hybrid_service.child_chunks),
        "total_parent_sections": len(hybrid_service.parent_sections),
        "knowledge_graph_nodes": hybrid_service.graph_store.graph.number_of_nodes(),
        "knowledge_graph_edges": hybrid_service.graph_store.graph.number_of_edges(),
    }


@router.delete("/documents")
def clear_documents():
    """Clears all indexed chunks and graph entities."""
    hybrid_service.child_chunks = []
    hybrid_service.parent_sections = {}
    hybrid_service.graph_store.clear()
    return {"status": "cleared"}
