"""Phase 14 - Evaluation metrics for the HulioMed RAG pipeline.

Runs labeled gold cases through the pipeline (using the Phase 9-13 code
unchanged) and computes simple, transparent metrics:

  Retrieval
    - hit@4       : did the top-4 retrieved chunks overlap the true pages?
    - gate_acc    : did the relevance decision match the label?
    - margin      : distance gap between the closest relevant and the closest
                    out-of-scope query (higher = easier to separate)

  Guardrails
    - block_recall: of the questions that MUST be blocked, how many were?
    - precision   : of questions that MUST be allowed, how many were?
    - over_block  : benign questions wrongly blocked (want 0)

  Answers (LlM is nondeterministic, so this is a small sample)
    - answer_rate : of fact cases, how many were answered+verified?
    - fact_rate   : how many answers stated the expected fact?
    - grounding   : how many facts were present verbatim in the context?
    - citation    : how many verified answers carried a page citation?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve as rt  # noqa: E402
from src.guardrails import guardrails as grd  # noqa: E402


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def evaluate_retrieval(cases):
    per_case = []
    for case in cases:
        out = rt.retrieve(case["query"], top_k=4)
        hit_pages = set()
        for hit in out["hits"]:
            hit_pages.update(hit["pages"])
        relevant = bool(case["gold_pages"])
        hit = bool(hit_pages & set(case["gold_pages"]))
        decision_ok = out["relevant"] == case["gold_relevant"]
        per_case.append({
            "query": case["query"],
            "gold_relevant": case["gold_relevant"],
            "top_distance": out["top_distance"],
            "hit": hit,
            "gold_pages": case["gold_pages"],
            "hit_pages": sorted(hit_pages),
            "decision_ok": decision_ok,
            # only meaningful for relevant cases
            "relevant": relevant,
        })

    relevant_cases = [c for c in per_case if c["relevant"]]
    irrelevant_cases = [c for c in per_case if not c["relevant"]]

    hit_rate = sum(c["hit"] for c in relevant_cases) / max(len(relevant_cases), 1)
    gate_acc = sum(c["decision_ok"] for c in per_case) / max(len(per_case), 1)

    max_relevant = max((c["top_distance"] for c in relevant_cases), default=None)
    min_irrelevant = min((c["top_distance"] for c in irrelevant_cases),
                         default=None)
    margin = min_irrelevant - max_relevant if (
        max_relevant is not None and min_irrelevant is not None
    ) else None

    return {
        "per_case": per_case,
        "hit_rate": hit_rate,
        "gate_accuracy": gate_acc,
        "margin": margin,
        "n_relevant": len(relevant_cases),
        "n_irrelevant": len(irrelevant_cases),
    }


# ---------------------------------------------------------------------------
# Guardrails (regex-only: free, deterministic, instant)
# ---------------------------------------------------------------------------
def evaluate_guardrails(cases):
    per_case = []
    block_recall_hits = 0
    block_recall_total = 0
    allowed_ok = 0
    allowed_total = 0
    for case in cases:
        checks = grd.check_input(case["query"])
        actual = grd.first_blocked(checks)
        expected = case["expected"]
        ok = (expected == "allowed" and actual is None) or (actual == expected)
        per_case.append({
            "query": case["query"],
            "expected": expected,
            "actual": actual,
            "ok": ok,
        })
        if expected == "allowed":
            allowed_total += 1
            if actual is None:
                allowed_ok += 1
        else:
            block_recall_total += 1
            if actual == expected:
                block_recall_hits += 1

    n_categories = {
        c: sum(1 for x in per_case if x["expected"] == c)
        for c in ("prompt_injection", "personal", "emergency", "pii")
    }
    category_recall = {}
    for cat, total in n_categories.items():
        hits = sum(
            1 for x in per_case if x["expected"] == cat and x["actual"] == cat
        )
        category_recall[cat] = hits / max(total, 1)

    return {
        "per_case": per_case,
        "block_recall": block_recall_hits / max(block_recall_total, 1),
        "precision": allowed_ok / max(allowed_total, 1),
        "over_block": allowed_total - allowed_ok,
        "misses": block_recall_total - block_recall_hits,
        "category_recall": category_recall,
        "n_block": block_recall_total,
        "n_allowed": allowed_total,
    }


# ---------------------------------------------------------------------------
# Answers (needs the LLM: small sample, spaced out for the free tier)
# ---------------------------------------------------------------------------
def evaluate_answers(cases, sleep_seconds=2.5):
    import time

    per_case = []
    n_answered = n_fact = n_grounded = n_citation = 0
    for case in cases:
        response = grd.answer_safely(case["query"])
        time.sleep(sleep_seconds)

        answered = response["status"] == "answered" and response["verified"] is True
        answer = response["answer"].lower()
        fact = case["fact"].lower()
        fact_ok = answered and fact in answer
        citation_ok = answered and "page" in answer and bool(response["sources"])

        context = " ".join(
            h["text"] for h in rt.retrieve(case["query"])["hits"]
        )
        grounding_ok = fact_ok and fact in context.lower()

        per_case.append({
            "query": case["query"],
            "fact": case["fact"],
            "answered": answered,
            "fact_ok": fact_ok,
            "grounding_ok": grounding_ok,
            "citation_ok": citation_ok,
            "answer_preview": response["answer"].replace("\n", " ")[:120],
        })

        n_answered += int(answered)
        n_fact += int(fact_ok)
        n_grounded += int(grounding_ok)
        n_citation += int(citation_ok)

    total = max(len(per_case), 1)
    return {
        "per_case": per_case,
        "n": len(per_case),
        "answer_rate": n_answered / total,
        "fact_rate": n_fact / total,
        "grounding_rate": n_grounded / total,
        "citation_rate": n_citation / total,
    }


def main_report(cases):
    retrieval = evaluate_retrieval(cases["retrieval"])
    guardrails = evaluate_guardrails(cases["guardrails"])
    answers = evaluate_answers(cases["answers"])
    return {"retrieval": retrieval, "guardrails": guardrails, "answers": answers}


if __name__ == "__main__":
    import json
    import time as _time

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = json.loads(
        (PROJECT_ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8")
    )
    report = main_report(dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))