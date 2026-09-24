"""Phase 9 - Retrieval: turn a user question into the most relevant monograph
chunks.

Steps:
  1. Embed the question with the SAME Mistral model used in Phase 6.
  2. Ask ChromaDB for the top-k closest chunks (by cosine distance).
  3. Judge relevance: if the best match is still too far away, the question is
     considered NOT answerable from the monograph (relevant=False).

The return value drives Phase 10 (answer generation) and Phase 11 (guardrails):
  - relevant=True  -> hand the hits to the LLM as the only context.
  - relevant=False -> the app will say the info cannot be verified.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "hulio_monograph"
EMBEDDING_MODEL = "mistral-embed"
TOP_K = 4                # how many chunks to fetch by default
RELEVANCE_THRESHOLD = 0.30  # best cosine distance below this = relevant

_collection_cache = None


def get_collection():
    """Return the ChromaDB collection (opened once, then reused)."""
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection_cache = client.get_collection(COLLECTION_NAME)
    return _collection_cache


def get_embed_client():
    """Read the Mistral API key from .env and build a client."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("MISTRAL_API_KEY missing in .env")
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral
    return Mistral(api_key=api_key)


def embed_query(text: str) -> list:
    """Embed a single question into a vector (same model as Phase 6)."""
    client = get_embed_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=[text])
    ordered = sorted(response.data, key=lambda item: item.index)
    return ordered[0].embedding


def parse_pages(pages_string: str) -> list:
    """Turn a stored '1,2,3' string back into a list of integers."""
    return [int(p) for p in pages_string.split(",") if p.strip()]


def retrieve(query: str, top_k: int = TOP_K,
             threshold: float = RELEVANCE_THRESHOLD) -> dict:
    """Return the top-k relevant chunks for a question, plus a relevance flag."""
    query_vector = embed_query(query)

    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=min(top_k, 4),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        hits.append(
            {
                "chunk_id": chunk_id,
                "document": meta["document"],
                "pages": parse_pages(meta["pages"]),
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "distance": results["distances"][0][i],
                "text": results["documents"][0][i],
            }
        )

    hits.sort(key=lambda h: h["distance"])  # closest meaning first
    top_distance = hits[0]["distance"] if hits else None

    return {
        "query": query,
        "relevant": top_distance is not None and top_distance <= threshold,
        "top_distance": top_distance,
        "threshold": threshold,
        "hits": hits,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python retrieve.py \"your question here\"")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(__import__("json").dumps(
        retrieve(" ".join(sys.argv[1:])),
        ensure_ascii=False,
        indent=2,
    ))