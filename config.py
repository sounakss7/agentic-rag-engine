import os
from typing import Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

def get_secret(key_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves a secret preferentially from Streamlit Secrets (if running in Streamlit),
    falling back to os.environ.
    """
    # Try reading from streamlit secrets safely
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = st.secrets[key_name]
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    
    # Fallback to environment variables
    env_val = os.getenv(key_name, default)
    if env_val and str(env_val).strip():
        return str(env_val).strip()
    
    return default


@dataclass
class AppConfig:
    """Central configuration for Advanced CRAG Engine."""
    # API Keys
    GEMINI_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None

    # LLM & Embeddings Settings
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"
    EMBEDDING_DIM: int = 768

    # Chunking Parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Retrieval & Reranking Settings
    QDRANT_COLLECTION: str = "crag_knowledge_base"
    RRF_K: int = 60
    TOP_K_SPARSE: int = 10
    TOP_K_DENSE: int = 10
    TOP_N_RERANK: int = 3
    FLASHRANK_MODEL: str = "ms-marco-MiniLM-L-6-v2"

    # CRAG Evaluation Thresholds
    RELEVANCE_THRESHOLD: float = 0.6

    def refresh_keys(self) -> None:
        """Loads or updates keys from secrets / env."""
        self.GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
        self.QDRANT_URL = get_secret("QDRANT_URL")
        self.QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
        self.TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
        self.LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")

    def key_status(self) -> Dict[str, bool]:
        """Returns status map of key availability."""
        self.refresh_keys()
        return {
            "GEMINI_API_KEY": bool(self.GEMINI_API_KEY),
            "QDRANT_URL": bool(self.QDRANT_URL),
            "QDRANT_API_KEY": bool(self.QDRANT_API_KEY),
            "TAVILY_API_KEY": bool(self.TAVILY_API_KEY),
            "LANGCHAIN_API_KEY": bool(self.LANGCHAIN_API_KEY),
        }


# Global Config Singleton
config = AppConfig()
config.refresh_keys()
