import os
from typing import Dict, Optional, List, Set, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Global Degraded Components Registry
DEGRADED_COMPONENTS: Set[str] = set()


def register_degraded_component(component_name: str) -> None:
    """Registers a component operating in degraded/stub/fallback mode."""
    DEGRADED_COMPONENTS.add(component_name)


def clear_degraded_component(component_name: str) -> None:
    """Removes a component from degraded registry."""
    DEGRADED_COMPONENTS.discard(component_name)


def get_degraded_components() -> List[str]:
    """Returns sorted list of active degraded/fallback component names."""
    return sorted(list(DEGRADED_COMPONENTS))


def is_degraded_mode() -> bool:
    """Returns True if any system component is currently running in degraded mode."""
    return len(DEGRADED_COMPONENTS) > 0


# Key Aliases mapping for flexible secret lookup
KEY_ALIASES: Dict[str, List[str]] = {
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key", "GEMINI_KEY", "google_key"],
    "QDRANT_URL": ["QDRANT_URL", "qdrant_url", "QDRANT_HOST"],
    "QDRANT_API_KEY": ["QDRANT_API_KEY", "qdrant_api_key"],
    "TAVILY_API_KEY": ["TAVILY_API_KEY", "tavily_api_key", "TAVILY_KEY"],
    "LANGCHAIN_API_KEY": ["LANGCHAIN_API_KEY", "langchain_api_key", "LANGSMITH_API_KEY"],
}


def get_secret(key_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Robust secret resolver:
    1. Checks Streamlit Secrets (st.secrets - top level & nested, case-insensitive, aliases)
    2. Checks environment variables (os.environ - aliases)
    """
    aliases = KEY_ALIASES.get(key_name, [key_name, key_name.lower()])

    # 1. Check Streamlit Secrets (st.secrets)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            # Direct alias lookup
            for alias in aliases:
                if alias in st.secrets and st.secrets[alias]:
                    val = str(st.secrets[alias]).strip()
                    if val:
                        return val

            # Case-insensitive top-level lookup
            sec_dict = dict(st.secrets)
            for alias in aliases:
                for k, v in sec_dict.items():
                    if str(k).upper() == alias.upper() and v:
                        val = str(v).strip()
                        if val:
                            return val

            # Check nested sections like [secrets], [default], or [env]
            for section in ["secrets", "default", "env"]:
                if section in st.secrets:
                    sec = st.secrets[section]
                    for alias in aliases:
                        if alias in sec and sec[alias]:
                            val = str(sec[alias]).strip()
                            if val:
                                return val
    except Exception:
        pass

    # 2. Check environment variables
    for alias in aliases:
        env_val = os.getenv(alias)
        if env_val and str(env_val).strip():
            return str(env_val).strip()

    return default


@dataclass
class AppConfig:
    """Central configuration for Advanced CRAG Engine."""
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

    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return get_secret("GEMINI_API_KEY")

    @property
    def QDRANT_URL(self) -> Optional[str]:
        return get_secret("QDRANT_URL")

    @property
    def QDRANT_API_KEY(self) -> Optional[str]:
        return get_secret("QDRANT_API_KEY")

    @property
    def TAVILY_API_KEY(self) -> Optional[str]:
        return get_secret("TAVILY_API_KEY")

    @property
    def LANGCHAIN_API_KEY(self) -> Optional[str]:
        return get_secret("LANGCHAIN_API_KEY")

    def refresh_keys(self) -> None:
        """Triggers dynamic check of secret status."""
        pass

    def key_status(self) -> Dict[str, bool]:
        """Returns status map of key availability."""
        return {
            "GEMINI_API_KEY": bool(self.GEMINI_API_KEY),
            "QDRANT_URL": bool(self.QDRANT_URL),
            "QDRANT_API_KEY": bool(self.QDRANT_API_KEY),
            "TAVILY_API_KEY": bool(self.TAVILY_API_KEY),
            "LANGCHAIN_API_KEY": bool(self.LANGCHAIN_API_KEY),
        }


# Global Config Singleton
config = AppConfig()
