"""Phase 8 - Full verification of the ChromaDB vector database.

Checks, in order:
  A. Collection identity and counts
  B. Document content  (stored text matches the chunk source, every chunk)
  C. Metadata          (document name, page strings, page bounds)
  D. Embeddings        (dimensions, finite values, normalization, fidelity)
  E. Search behaviour  (self-retrieval, semantic queries, out-of-scope query)

Uses only the stored vectors for all tests except the semantic queries, which
need one Mistral embedding call each (same model as Phase 6).
"""

import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "hulio_chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "hulio_embeddings.json"
EXTRACTED_PATH = PROJECT_ROOT / "data" / "hulio_monograph_extracted.json"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "hulio_monograph"
EXPECTED_DOCUMENT = "Hulio Product Monograph"
EXPECTED_DIMENSIONS = 1024
NORM_TOLERANCE = 1e-3
VECTOR_FIDELITY_TOLERANCE = 1e-4

results = []
passed = 0


def record(name: str, ok: bool, detail: str = "") -> None:
    """Add a named test result and print it as it runs."""
    global passed
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] {name}{suffix}")
    results.append((name, ok))
    if ok:
        passed += 1


def embed_one(text: str) -> list:
    """Embed a single string with Mistral's mistral-embed model."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("MISTRAL_API_KEY missing in .env")
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=[text])
    ordered = sorted(response.data, key=lambda item: item.index)
    return ordered[0].embedding


def vector_norm(vec: list) -> float:
    return math.sqrt(sum(v * v for v in vec))


def main() -> None:
    for path in (CHUNKS_PATH, EMBEDDINGS_PATH, EXTRACTED_PATH):
        if not path.exists():
            sys.exit(f"Required input missing: {path}")

    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        chunks = json.load(fh)
    with open(EMBEDDINGS_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    with open(EXTRACTED_PATH, encoding="utf-8") as fh:
        extracted = json.load(fh)

    records = payload["records"]
    total_pages = len(extracted)

    print("=== A. Collection identity and counts ===")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection_names = [c.name for c in client.list_collections()]
    record("A1 collection exists", COLLECTION_NAME in collection_names)
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    record("A2 count == 48 chunks", count == len(chunks) == 48,
           f"stored {count}")
    record("A3 count == embeddings records", count == len(records),
           f"embeddings {len(records)}")

    print("=== B. Document content (all 48 chunks) ===")
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    stored_map = {cid: i for i, cid in enumerate(stored["ids"])}
    text_ok = all(
        stored["documents"][stored_map[c["chunk_id"]]] == c["text"] for c in chunks
    )
    record("B1 all stored texts match chunk source exactly", text_ok)
    record("B2 no empty stored documents",
           all(bool(d) for d in stored["documents"]))

    print("=== C. Metadata (all 48 chunks) ===")
    meta_ok_doc = all(
        stored["metadatas"][stored_map[c["chunk_id"]]]["document"] == EXPECTED_DOCUMENT
        for c in chunks
    )
    record("C1 document name correct everywhere", meta_ok_doc)

    def pages_from_meta(meta) -> list:
        return [int(p) for p in meta["pages"].split(",") if p.strip()]

    pages_ok = all(
        pages_from_meta(stored["metadatas"][stored_map[c["chunk_id"]]]) == c["pages"]
        for c in chunks
    )
    record("C2 stored page strings match chunk pages", pages_ok)

    bounds_ok = all(
        (meta["page_start"] == pages[0] and meta["page_end"] == pages[-1])
        for c in chunks
        for meta in [stored["metadatas"][stored_map[c["chunk_id"]]]]
        for pages in [pages_from_meta(meta)]
    )
    record("C3 page_start/page_end agree with page strings", bounds_ok)

    flat_pages = [
        p
        for c in chunks
        for p in pages_from_meta(stored["metadatas"][stored_map[c["chunk_id"]]])
    ]
    record("C4 all page numbers within document 1..%d" % total_pages,
           min(flat_pages) >= 1 and max(flat_pages) <= total_pages,
           "range %d..%d" % (min(flat_pages), max(flat_pages)))

    print("=== D. Embeddings (all 48 vectors) ===")
    dims_ok = all(len(v) == EXPECTED_DIMENSIONS for v in stored["embeddings"])
    record("D1 every stored vector is %d-dimensional" % EXPECTED_DIMENSIONS,
           dims_ok)

    finite_ok = all(
        all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec)
        for vec in stored["embeddings"]
    )
    record("D2 all vector values finite", finite_ok)

    norms = [vector_norm(v) for v in stored["embeddings"]]
    record("D3 all vectors unit-length (within tol=%.0e)" % NORM_TOLERANCE,
           all(abs(n - 1.0) <= NORM_TOLERANCE for n in norms),
           "min %.4f max %.4f" % (min(norms), max(norms)))

    src_vec_map = {r["chunk_id"]: r["embedding"] for r in records}
    fidelity_issues = 0
    for cid, idx in stored_map.items():
        src_vec = src_vec_map[cid]
        got_vec = stored["embeddings"][idx]
        for a, b in zip(src_vec, got_vec):
            if not math.isclose(a, b, rel_tol=VECTOR_FIDELITY_TOLERANCE,
                                abs_tol=VECTOR_FIDELITY_TOLERANCE):
                fidelity_issues += 1
                break
    record("D4 stored vectors equal source embeddings (48/48, within tol)",
           fidelity_issues == 0, "mismatched: %d" % fidelity_issues)

    print("=== E. Search behaviour ===")
    # E1 self-retrieval: query each chunk with its OWN stored vector; the very
    # same chunk must be the #1 result (proves the vector->id mapping is intact).
    self_hits = 0
    for cid in stored["ids"]:
        res = collection.query(query_embeddings=[stored["embeddings"][stored_map[cid]]],
                               n_results=1)
        if res["ids"][0][0] == cid:
            self_hits += 1
    record("E1 self-retrieval: chunk finds itself as top result",
           self_hits == 48, "%d/48" % self_hits)

    # E2 semantic queries (real Mistral embeddings of the question).
    semantic_queries = [
        ("What dose of HULIO is recommended for plaque psoriasis?",
         "psoriasis", "contains psoriasis dosing text"),
        ("How should the HULIO prefilled pen be stored?",
         "storage", "mentions storage instructions"),
    ]
    for q, keyword, note in semantic_queries:
        qv = embed_one(q)
        res = collection.query(query_embeddings=[qv], n_results=3,
                               include=["documents", "metadatas", "distances"])
        top_doc = res["documents"][0][0]
        pages = res["metadatas"][0][0]["pages"]
        dist = res["distances"][0][0]

        keyword_hit = keyword.lower() in top_doc.lower()
        snippet = top_doc[:90].replace("\n", " ")
        record(f"E2 '{q}'",
               dist < 0.35,
               f"dist {dist:.3f}, pages [{pages}], hit='{keyword}'={keyword_hit}")
        print(f"        top: {snippet}...")

    # E3 out-of-scope question should be far from everything (large distance).
    oos_vec = embed_one("What will the weather be like tomorrow?")
    oos_res = collection.query(query_embeddings=[oos_vec], n_results=1,
                               include=["distances"])
    oos_dist = oos_res["distances"][0][0]
    record("E3 out-of-scope question is far from every chunk (informational)",
           oos_dist > 0.2, "best distance %.3f (larger = less relevant)" % oos_dist)

    print()
    print("=" * 60)
    print(f"SUMMARY: {passed}/{len(results)} checks passed")
    failed = [name for name, ok in results if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("VECTOR DATABASE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()