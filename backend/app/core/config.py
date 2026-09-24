"""
NexusRAG Core Configuration & Secret Management.
Robust multi-tier configuration using Pydantic Settings and dynamic secret resolution.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv(override=False)

# Track active degraded components
DEGRADED_REGISTRY: Dict[str, str] = {}

def register_degraded(component: str, reason: str = "") -> None:
    DEGRADED_REGISTRY[component] = reason

def clear_degraded(component: str) -> None:
    DEGRADED_REGISTRY.pop(component, None)

def get_degraded_status() -> Dict[str, str]:
    return dict(DEGRADED_REGISTRY)


class Settings(BaseSettings):
    """NexusRAG Global Enterprise Settings."""

    # Project metadata
    PROJECT_NAME: str = "NexusRAG - Enterprise Autonomous Deep-Research & Graph-RAG Engine"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False

    # LLM & Generation Models
    PRIMARY_LLM: str = "gemini-2.5-flash"
    CASCADE_MODELS: List[str] = Field(
        default_factory=lambda: ["gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]
    )
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768

    # Chunking & Hierarchical Ingestion
    PARENT_CHUNK_SIZE: int = 1200
    CHILD_CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 40

    # Retrieval & Ranking Parameters
    QDRANT_COLLECTION: str = "nexus_knowledge_base"
    RRF_K: int = 60
    TOP_K_VECTOR: int = 10
    TOP_K_SPARSE: int = 10
    TOP_K_GRAPH: int = 8
    TOP_N_RERANK: int = 4
    FLASHRANK_MODEL: str = "ms-marco-TinyBERT-L-2-v2"
    RELEVANCE_THRESHOLD: float = 0.60

    # API Keys & Endpoints
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None

    class Config:
        case_sensitive = True
        extra = "allow"

    @property
    def active_gemini_key(self) -> Optional[str]:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    @property
    def active_qdrant_url(self) -> Optional[str]:
        return self.QDRANT_URL or os.getenv("QDRANT_URL")

    @property
    def active_qdrant_key(self) -> Optional[str]:
        return self.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY")

    @property
    def active_tavily_key(self) -> Optional[str]:
        return self.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")

    @property
    def active_langchain_key(self) -> Optional[str]:
        return self.LANGCHAIN_API_KEY or os.getenv("LANGCHAIN_API_KEY")

    def key_status(self) -> Dict[str, bool]:
        return {
            "GEMINI_API_KEY": bool(self.active_gemini_key),
            "QDRANT_URL": bool(self.active_qdrant_url),
            "QDRANT_API_KEY": bool(self.active_qdrant_key),
            "TAVILY_API_KEY": bool(self.active_tavily_key),
            "LANGCHAIN_API_KEY": bool(self.active_langchain_key),
        }


# Global Singleton Settings Instance
settings = Settings()
