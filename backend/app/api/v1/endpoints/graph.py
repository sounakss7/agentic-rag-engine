"""
Knowledge Graph visualization endpoints.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints.documents import hybrid_service

router = APIRouter()

@router.get("/visualize")
def get_graph_data(limit: int = 80):
    """Returns nodes and edges for Cytoscape / D3 graph rendering."""
    return hybrid_service.graph_store.get_visualization_data(max_nodes=limit)
