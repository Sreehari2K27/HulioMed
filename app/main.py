"""Phase 12 - HulioMed backend API (FastAPI).

Exposes the full protected RAG pipeline (Phases 9-11) through two endpoints:

  GET  /health  -> liveness + system info
  POST /ask     -> one protected answer for one question

Design notes:
- No chat history is persisted anywhere (PII safety requirement).
- Guardrail outcomes (blocked / fallback / answered) are NOT HTTP errors:
  they are all returned as HTTP 200 with a `status` field, so the UI has one
  simple contract.
- Request/response payloads are validated with Pydantic; an invalid request
  gets HTTP 422 automatically.
- Run: uvicorn app.main:app --reload   (from the project root)
"""

import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.guardrails import answer_safely  # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic models (validation layer)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown fields (422)

    question: str = Field(
        min_length=1,
        max_length=500,
        description="A question about HULIO (adalimumab-fkjp).",
        examples=["What is the recommended dose of HULIO for plaque psoriasis?"],
    )

    @field_validator("question")
    @classmethod
    def strip_and_require_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty or whitespace")
        return value


class Source(BaseModel):
    chunk_id: str
    document: str
    pages: list[int]
    distance: float


class Check(BaseModel):
    check: str
    passed: bool
    detail: str = ""


class Checks(BaseModel):
    input: list[Check] = []
    output: list[Check] = []


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "answered_out_of_scope", "blocked"]
    blocked_reason: str | None = None
    verified: bool
    answer: str
    sources: list[Source] = []
    checks: Checks


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HulioMed API",
    description=(
        "Medical-information assistant for HULIO (adalimumab-fkjp). Answers "
        "are grounded only in the Hulio Product Monograph, with page "
        "citations, safety guardrails, and fallbacks."
    ),
    version="1.0.0",
)

# CORS for the demo. By default any origin is allowed (educational prototype,
# no credentials, no auth). To restrict, set CORS_ORIGINS to a comma-separated
# list of exact origins, e.g. CORS_ORIGINS="https://demo.streamlit.app".
_cors_env = os.getenv("CORS_ORIGINS", "*").strip()
allow_origins = [
    origin.strip() for origin in _cors_env.split(",") if origin.strip()
] if _cors_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "name": "HulioMed API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/ask"],
    }


@app.get("/health")
def health():
    info = {
        "status": "ok",
        "collection": "hulio_monograph",
        "embedding_model": "mistral-embed",
        "llm_model": "open-mistral-nemo",
        "chunks": None,
    }
    try:
        from src.retrieval.retrieve import get_collection

        info["chunks"] = get_collection().count()
    except Exception as exc:  # noqa: BLE001 - report degraded, don't crash
        info["status"] = "degraded"
        info["error"] = str(exc)
    return info


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Run one protected question through the full RAG pipeline."""
    return answer_safely(request.question)