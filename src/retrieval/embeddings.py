"""Phase 6 - Convert each chunk into an embedding (a list of numbers).

Reads data/hulio_chunks.json and, for every chunk, calls the Mistral embedding
model to get a vector that captures the chunk's meaning. Saves the result to
data/hulio_embeddings.json so Phase 7 can store these vectors in ChromaDB.

Why embeddings? Two pieces of text with similar MEANING get vectors that are
close together, even if they use completely different words. That is what lets
the chatbot later match "how much do I inject?" to "recommended subcutaneous
dosage".
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "hulio_chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hulio_embeddings.json"

EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIMENSIONS = 1024


def get_client():
    """Read the API key from .env and build a Mistral client."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or "your-mistral-api-key-here" in api_key:
        sys.exit(
            "MISTRAL_API_KEY not found.\n"
            "1) Copy .env.example to .env  (or edit the existing .env)\n"
            "2) Paste your real Mistral API key into .env\n"
            "3) Re-run this script."
        )
    try:
        from mistralai.client import Mistral  # mistralai SDK v2.x
    except ImportError:
        from mistralai import Mistral  # mistralai SDK v1.x
        return Mistral(api_key=api_key)
    return Mistral(api_key=api_key)


def embed_chunks(client, texts: list) -> list:
    """Embed a batch of texts in one API call and return their vectors."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def main() -> None:
    if not CHUNKS_PATH.exists():
        sys.exit(f"Chunks JSON not found: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        chunks = json.load(fh)

    texts = [chunk["text"] for chunk in chunks]

    client = get_client()
    vectors = embed_chunks(client, texts)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "pages": chunk["pages"],
                "text": chunk["text"],
                "embedding": vector,
            }
        )

    payload = {
        "embedding_provider": "Mistral",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_count": len(records),
        "records": records,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Chunks embedded:     {len(records)}")
    print(f"Embedding model:     {EMBEDDING_MODEL}")
    print(f"Vector dimensions:   {EMBEDDING_DIMENSIONS}")
    print(f"JSON written to:     {OUTPUT_PATH}")


if __name__ == "__main__":
    main()