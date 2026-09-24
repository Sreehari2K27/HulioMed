"""Phase 5 - Verify the chunking output.

Checks that chunks exist, IDs and page metadata are valid, chunk sizes are
reasonable, and the output JSON was created correctly.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "data" / "hulio_chunks.json"

MIN_WORDS = 40
MAX_WORDS = 900
REQUIRED_KEYS = {"chunk_id", "document", "pages", "text"}


def check(ok: bool, message: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {message}")
    return ok


def main() -> None:
    all_passed = True

    if not JSON_PATH.exists():
        check(False, f"Chunk JSON not found at {JSON_PATH}")
        sys.exit(1)
    all_passed = check(True, "Chunk JSON file exists") and all_passed

    with open(JSON_PATH, encoding="utf-8") as fh:
        chunks = json.load(fh)

    all_passed = check(isinstance(chunks, list) and len(chunks) > 0,
                       f"JSON contains a non-empty list of {len(chunks)} chunks") and all_passed

    all_passed = check(all(REQUIRED_KEYS <= set(c.keys()) for c in chunks),
                       "every chunk has chunk_id, document, pages and text") and all_passed

    ids = [c["chunk_id"] for c in chunks]
    all_passed = check(len(ids) == len(set(ids)), "every chunk has a unique chunk_id") and all_passed

    all_passed = check(all(bool(c["text"]) for c in chunks), "every chunk has text") and all_passed

    all_passed = check(all(isinstance(c["pages"], list) and c["pages"] for c in chunks),
                       "every chunk has non-empty page metadata") and all_passed
    all_passed = check(all(all(isinstance(p, int) for p in c["pages"]) for c in chunks),
                       "page metadata values are numbers") and all_passed

    all_passed = check(all(c["document"] == "Hulio Product Monograph" for c in chunks),
                       "document name is 'Hulio Product Monograph' on all chunks") and all_passed

    sizes = [len(c["text"].split()) for c in chunks]
    in_range = all(MIN_WORDS <= size <= MAX_WORDS for size in sizes)
    all_passed = check(in_range,
                       f"all chunks are {MIN_WORDS}-{MAX_WORDS} words (actual min {min(sizes)}, max {max(sizes)})") and all_passed

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()