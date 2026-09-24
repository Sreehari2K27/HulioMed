"""Phase 9 - Verify the retrieval logic.

Checks:
  A. Relevant questions are flagged relevant and return monograph chunks.
  B. Known answers actually appear in the retrieved chunks.
  C. Out-of-scope questions are flagged NOT relevant (fallback trigger).
  D. Results are ranked by distance (closest first).
  E. Metadata (pages, document) is always present and valid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval import retrieve as rt

EXPECTED = [
    ("What is the recommended dose for plaque psoriasis?",
     True, ["80 mg", "Plaque Psoriasis"]),
    ("How is HULIO dosed for ulcerative colitis in adults?",
     True, ["160 mg", "ulcerative colitis"]),
    ("What dose should a child with Crohn's disease receive?",
     True, ["pediatric"]),
    ("How should the HULIO prefilled pen be stored?",
     True, ["frozen"]),
    ("Which conditions is HULIO indicated for?",
     True, ["indicated for"]),
    ("What are the serious side effects of adalimumab?",
     True, ["infections"]),
    ("Can HULIO be used during pregnancy?",
     True, ["Pregnancy"]),
]

OUT_OF_SCOPE = [
    ("What is the weather like in Mumbai today?", False),
    ("Who won the cricket world cup in 2023?", False),
    ("Tell me how to file my income tax returns.", False),
]

results = []
passed = 0


def record(name: str, ok: bool, detail: str = "") -> None:
    global passed
    status = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] {name}{suffix}")
    results.append((name, ok))
    if ok:
        passed += 1


def main() -> None:
    print("=== A & B. Relevant questions + content checks ===")
    for query, expect_relevant, snippets in EXPECTED:
        out = rt.retrieve(query, top_k=4)
        ok_flag = out["relevant"] == expect_relevant
        ok_hits = len(out["hits"]) > 0
        joined = " ".join(h["text"] for h in out["hits"])
        missing = [s for s in snippets if s.lower() not in joined.lower()]
        ok_content = not missing
        top_pages = out["hits"][0]["pages"]
        record(
            f"'{query[:48]}'",
            ok_flag and ok_hits and ok_content,
            "dist=%.3f pages=%s missing=%s"
            % (out["top_distance"], top_pages, missing or "none"),
        )

    print("=== C. Out-of-scope questions trigger the fallback ===")
    for query, expect in OUT_OF_SCOPE:
        out = rt.retrieve(query, top_k=4)
        record(
            f"'{query[:48]}' -> relevant={expect}",
            out["relevant"] == expect,
            "dist=%.3f" % out["top_distance"],
        )

    print("=== D. Ranking by distance (all queries combined) ===")
    all_queries = [q for q, _, _ in EXPECTED] + [q for q, _ in OUT_OF_SCOPE]
    ranked_ok = True
    for query in all_queries:
        out = rt.retrieve(query, top_k=4)
        dists = [h["distance"] for h in out["hits"]]
        if dists != sorted(dists):
            ranked_ok = False
            print(f"  unranked: {query[:40]} {dists}")
    record("hits are sorted closest-first for every query", ranked_ok)

    print("=== E. Metadata validity (all queries) ===")
    meta_ok = True
    pages_in_range = True
    for query in all_queries:
        for h in rt.retrieve(query, top_k=4)["hits"]:
            if h["document"] != "Hulio Product Monograph":
                meta_ok = False
            if not h["pages"] or min(h["pages"]) < 1 or max(h["pages"]) > 81:
                pages_in_range = False
    record("every hit carries correct document name", meta_ok)
    record("every hit carries valid page numbers within 1..81", pages_in_range)

    print()
    print("=" * 60)
    print(f"SUMMARY: {passed}/{len(results)} checks passed")
    failed = [n for n, ok in results if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("RETRIEVAL VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()