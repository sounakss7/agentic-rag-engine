"""
NexusRAG Chat & Deep-Research Endpoints.
Provides both standard JSON responses and real-time Server-Sent Events (SSE) streaming.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.api.v1.endpoints.documents import hybrid_service
from backend.app.services.agents.orchestrator import NexusOrchestrator

router = APIRouter()
orchestrator = NexusOrchestrator(hybrid_service)


class ChatRequest(BaseModel):
    query: str = Field(description="The user's question or research request")
    chat_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class CitationItem(BaseModel):
    citation_index: int
    source: str
    page: int
    snippet: str
    url: Optional[str] = ""


class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    source_type: str
    citations: List[CitationItem]
    code_generated: Optional[str] = None
    code_execution_result: Optional[str] = None
    critic_passed: bool
    critic_feedback: str
    node_trace: List[str]


@router.post("/chat", response_model=ChatResponse)
async def execute_chat(request: ChatRequest):
    """Executes the full NexusRAG multi-agent research workflow synchronously."""
    try:
        res = orchestrator.run(
            query=request.query,
            chat_history=request.chat_history
        )
        return ChatResponse(
            answer=res.get("final_answer", ""),
            confidence_score=res.get("confidence_score", 0.0),
            source_type=res.get("source_type", "NexusRAG Multi-Source"),
            citations=[CitationItem(**c) for c in res.get("citations", [])],
            code_generated=res.get("code_generated"),
            code_execution_result=res.get("code_execution_result"),
            critic_passed=res.get("critic_passed", True),
            critic_feedback=res.get("critic_feedback", ""),
            node_trace=res.get("node_trace", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research agent workflow failed: {e}")


@router.post("/chat/stream")
async def execute_chat_stream(request: ChatRequest):
    """
    Streams multi-agent thought steps, code execution logs, and final answer tokens via Server-Sent Events (SSE).
    """
    async def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'message': 'Initializing NexusRAG Orchestration Engine...'})}\n\n"
        await asyncio.sleep(0.05)

        # Run orchestrator in threadpool to keep event loop responsive
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            orchestrator.run,
            request.query,
            request.chat_history
        )

        # Stream traces sequentially
        for step in res.get("node_trace", []):
            yield f"data: {json.dumps({'event': 'trace', 'step': step})}\n\n"
            await asyncio.sleep(0.04)

        # Stream code execution if present
        if res.get("code_execution_result"):
            yield f"data: {json.dumps({'event': 'code_execution', 'code': res.get('code_generated'), 'output': res.get('code_execution_result')})}\n\n"
            await asyncio.sleep(0.05)

        # Stream final answer
        payload = {
            "event": "answer",
            "answer": res.get("final_answer", ""),
            "confidence_score": res.get("confidence_score", 0.0),
            "source_type": res.get("source_type", "NexusRAG Multi-Source"),
            "citations": res.get("citations", []),
            "critic_passed": res.get("critic_passed", True)
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
