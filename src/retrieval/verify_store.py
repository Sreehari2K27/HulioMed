"""Phase 7 - Verify the ChromaDB collection and run a similarity search test.

Checks the collection exists and holds one vector per chunk, spot-checks that
retrieved rows match the source data, and embeds a real question to confirm the
collection can be searched by meaning.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "hulio_embeddings.json"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "hulio_monograph"


def check(ok: bool, message: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")
    return ok


def embed_query(query: str) -> list:
    """Embed the test question using the same Mistral embedding model."""
    load_dotenv(PROJECT_ROOT / ".env")
    import os

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("MISTRAL_API_KEY missing in .env")
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    ordered = sorted(response.data, key=lambda item: item.index)
    return ordered[0].embedding


def main() -> None:
    all_passed = True

    if not EMBEDDINGS_PATH.exists():
        sys.exit(f"Embeddings JSON not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload["records"]
    expected_ids = [r["chunk_id"] for r in records]

    client = chromadb.PersistentClient(path=str(DB_PATH))
    all_passed = check(COLLECTION_NAME in [c.name for c in client.list_collections()],
                       f"collection '{COLLECTION_NAME}' exists") and all_passed

    collection = client.get_collection(COLLECTION_NAME)
    stored_count = collection.count()
    all_passed = check(stored_count == len(expected_ids) == 48,
                       f"collection count {stored_count} matches 48 chunks") and all_passed

    stored = collection.get(include=["documents", "metadatas"])
    stored_ids = stored["ids"]
    all_passed = check(len(stored_ids) == len(set(stored_ids)),
                       "stored IDs are unique") and all_passed
    all_passed = check(sorted(stored_ids) == sorted(expected_ids),
                       "stored IDs match the chunk IDs") and all_passed

    # Spot-check three known chunks against the source JSON.
    spot_ids = ["hulio_0001", "hulio_0002", "hulio_0048"]
    ok_spot = True
    for cid in spot_ids:
        idx = stored_ids.index(cid)
        src = next(r for r in records if r["chunk_id"] == cid)
        text_ok = stored["documents"][idx] == src["text"]
        pages_ok = stored["metadatas"][idx]["pages"] == ",".join(str(p) for p in src["pages"])
        doc_ok = stored["metadatas"][idx]["document"] == src["document"]
        ok_spot = ok_spot and text_ok and pages_ok and doc_ok
        print(f"  spot-check {cid}: text_match={text_ok} pages_match={pages_ok} document_match={doc_ok}")
    all_passed = check(ok_spot, "spot-checked chunks match source data exactly") and all_passed

    print()
    print("SIMILARITY SEARCH TEST")
    query = "What is the recommended dose of HULIO for plaque psoriasis?"
    qv = embed_query(query)
    results = collection.query(query_embeddings=[qv], n_results=3, include=["documents", "metadatas", "distances"])

    print(f"query: {query}")
    for i, cid in enumerate(results["ids"][0]):
        dist = results["distances"][0][i]
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        print(f"\n  rank {i+1}  id={cid}  pages=[{meta['pages']}]  distance={dist:.4f}")
        print(f"  text: {doc[:220]}...")

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()