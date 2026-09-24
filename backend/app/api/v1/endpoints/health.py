"""
Health check and telemetry endpoint.
"""

from fastapi import APIRouter
from backend.app.core.config import settings, get_degraded_status

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "key_status": settings.key_status(),
        "degraded_components": get_degraded_status()
    }
