"""
NexusRAG API v1 Central Router.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, documents, graph, chat

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Diagnostics"])
api_router.include_router(documents.router, tags=["Documents & Ingestion"])
api_router.include_router(graph.router, tags=["Knowledge Graph"])
api_router.include_router(chat.router, tags=["Multi-Agent Research Chat"])
