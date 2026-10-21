"""PaperPilot — FastAPI service.

Serves the same retrieve-then-generate pipeline as the Streamlit UI (and,
optionally, the Claude agent) over HTTP, returning grounded, page-cited
answers as JSON.

Usage:
    uvicorn src.api:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \\
         -d '{"question": "How are radioactive materials transported?"}'
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import DEFAULT_INDEX_DIR, DEFAULT_K, DEFAULT_LLM, RAGPipeline

app = FastAPI(title="PaperPilot API", version="1.0")

# One model instance serves all requests; generation isn't thread-safe.
_generate_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(DEFAULT_K, ge=1, le=10, description="chunks to retrieve")
    mode: str = Field("few_shot", pattern="^(zero_shot|few_shot)$")
    backend: str = Field("local", pattern="^(local|agentic)$")


class Source(BaseModel):
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    backend: str
    retrieval_ms: float | None = None
    generation_s: float | None = None
    tool_calls: list[dict] = []
    truncated: bool = False


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Load the pipeline once, on first use."""
    model = os.environ.get("PAPERPILOT_LLM", DEFAULT_LLM)
    return RAGPipeline(index_dir=DEFAULT_INDEX_DIR, model_name=model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    if req.backend == "agentic":
        return _ask_agentic(req, pipeline)

    with _generate_lock:
        result = pipeline.answer(req.question, k=req.k, mode=req.mode)
    return AskResponse(
        answer=result.answer,
        sources=[Source(page=c.page, score=c.score, text=c.text) for c in result.contexts],
        backend="local",
        retrieval_ms=result.retrieval_latency * 1000,
        generation_s=result.generation_latency,
    )


def _ask_agentic(req: AskRequest, pipeline: RAGPipeline) -> AskResponse:
    # Imported here so the local backend works without the anthropic package
    # being configured.
    import anthropic

    from src.agent import answer as agent_answer
    from src.agent import is_missing_credential_error

    try:
        result = agent_answer(req.question, pipeline.retriever)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "ANTHROPIC_API_KEY was rejected by the API.")
    except TypeError as e:
        if not is_missing_credential_error(e):
            raise
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set; agentic backend unavailable.")
    return AskResponse(
        answer=result.answer,
        backend="agentic",
        tool_calls=result.tool_calls,
        truncated=result.truncated,
    )
