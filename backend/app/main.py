"""
NexusRAG FastAPI Enterprise Application Gateway.
Provides asynchronous endpoints, SSE streaming, OpenAPI docs, and CORS middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing NexusRAG Engine...")
    logger.info(f"Primary LLM: {settings.PRIMARY_LLM}, Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"API Key Status: {settings.key_status()}")
    yield
    logger.info("NexusRAG Engine shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Autonomous Research & Graph-RAG Engine with Layout-Aware Ingestion, Python Sandbox, and Multi-Agent Orchestration.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "engine": settings.PROJECT_NAME,
        "version": "2.0.0",
        "documentation": "/docs",
        "health": f"{settings.API_V1_STR}/health",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
