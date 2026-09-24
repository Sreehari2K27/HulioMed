"""Phase 13 - tiny HTTP client: the chat UI talks ONLY to the FastAPI backend.

The UI never touches ChromaDB or the model directly - every question goes to
POST /ask (Phase 12), which runs the full protected RAG pipeline and returns
a structured response.
"""

import httpx

DEFAULT_TIMEOUT = 90.0  # generous: one Mistral embed + one chat call


def health(base_url: str, timeout: float = 10.0):
    """Return the backend's /health payload, or None if unreachable."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


def ask(question: str, base_url: str, timeout: float = DEFAULT_TIMEOUT):
    """Send one question to the backend; return its JSON or None on failure."""
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/ask",
            json={"question": question},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None