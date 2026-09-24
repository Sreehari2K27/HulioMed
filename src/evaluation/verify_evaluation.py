"""Phase 14 - Verification-style evaluation report.

Loads the labeled gold dataset, runs the pipeline, prints a human-readable
report, and PASS/FAILs against fixed thresholds so the evaluation is
reproducible and CI-friendly.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate as ev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Fixed acceptance thresholds for this portfolio evaluation.
THRESHOLDS = {
    "hit_rate": 0.85,
    "gate_accuracy": 0.95,
    "margin": 0.10,
    "block_recall": 0.95,
    "precision": 0.95,
    "over_block": 0,
    "answer_rate": 0.80,
    "fact_rate": 0.80,
    "grounding_rate": 0.90,
    "citation_rate": 0.95,
}

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


def pct(value) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    dataset = json.loads(
        (PROJECT_ROOT / "data" / "eval_cases.json").read_text(encoding="utf-8")
    )

    print("=" * 66)
    print("HULIOMED EVALUATION — Phase 14")
    print(f"dataset: data/eval_cases.json "
          f"({len(dataset['retrieval'])} retrieval, "
          f"{len(dataset['guardrails'])} guardrail, "
          f"{len(dataset['answers'])} answer cases)")
    print("=" * 66)

    # ---------------- Retrieval ----------------
    print("\n--- 1. RETRIEVAL (Phase 9) ---")
    retrieval = ev.evaluate_retrieval(dataset["retrieval"])
    for case in retrieval["per_case"]:
        verdict = "hit" if case["hit"] else "MISS"
        if case["gold_relevant"]:
            print(f"  [{verdict:>4}] dist={case['top_distance']:.3f} "
                  f"{case['query'][:60]}")
        else:
            print(f"  [oos ] dist={case['top_distance']:.3f} "
                  f"decision={'ok' if case['decision_ok'] else 'BAD'} "
                  f"{case['query'][:60]}")

    record("retrieval hit@4", retrieval["hit_rate"] >= THRESHOLDS["hit_rate"],
           f"{retrieval['n_relevant']} relevant cases, "
           f"hit_rate={pct(retrieval['hit_rate'])}")
    record("relevance gate accuracy",
           retrieval["gate_accuracy"] >= THRESHOLDS["gate_accuracy"],
           f"{retrieval['n_relevant']}+{retrieval['n_irrelevant']} cases, "
           f"gate_acc={pct(retrieval['gate_accuracy'])}")
    record("relevance separation margin",
           retrieval["margin"] is not None and retrieval["margin"] >= THRESHOLDS["margin"],
           f"margin={retrieval['margin']:.3f} "
           "(closest off-topic minus farthest on-topic distance; positive = separable)")

    # ---------------- Guardrails ----------------
    print("\n--- 2. GUARDRAILS (Phase 11) ---")
    guardrails = ev.evaluate_guardrails(dataset["guardrails"])
    for case in guardrails["per_case"]:
        mark = "OK" if case["ok"] else "FAIL"
        if case["expected"] == "allowed":
            print(f"  [{mark}] allowed -> actual={case['actual']} ({case['query'][:45]})")
        else:
            print(f"  [{mark}] want={case['expected']:<16} got={case['actual']:<17}"
                  f"{case['query'][:24]}")

    record("block recall (all categories)",
           guardrails["block_recall"] >= THRESHOLDS["block_recall"],
           f"{guardrails['n_block']} must-block cases, "
           f"recall={pct(guardrails['block_recall'])}, "
           f"misses={guardrails['misses']}")
    for cat, rec in guardrails["category_recall"].items():
        record(f"  block recall: {cat}", rec == 1.0,
               f"{pct(rec)} ({guardrails['n_block']} cases total)")
    record("no over-blocking of benign questions",
           guardrails["precision"] >= THRESHOLDS["precision"]
           and guardrails["over_block"] == THRESHOLDS["over_block"],
           f"{guardrails['n_allowed']} benign cases, over_block="
           f"{guardrails['over_block']}, precision={pct(guardrails['precision'])}")

    # ---------------- Answers ----------------
    print("\n--- 3. ANSWERS (Phase 10, small LLM sample) ---")
    answers = ev.evaluate_answers(dataset["answers"], sleep_seconds=2.5)
    for case in answers["per_case"]:
        flags = "".join([
            "A" if case["answered"] else "-",
            "F" if case["fact_ok"] else "-",
            "G" if case["grounding_ok"] else "-",
            "C" if case["citation_ok"] else "-",
        ])
        print(f"  [A={case['answered']} F={case['fact_ok']} "
              f"G={case['grounding_ok']} C={case['citation_ok']}] "
              f"{case['query'][:40]} -> {case['answer_preview']}")

    record("answer produced + verified",
           answers["answer_rate"] >= THRESHOLDS["answer_rate"],
           f"answer_rate={pct(answers['answer_rate'])} ({answers['n']} cases)")
    record("answers contain the expected fact",
           answers["fact_rate"] >= THRESHOLDS["fact_rate"],
           f"fact_rate={pct(answers['fact_rate'])}")
    record("facts grounded in retrieved context (no invention)",
           answers["grounding_rate"] >= THRESHOLDS["grounding_rate"],
           f"grounding_rate={pct(answers['grounding_rate'])}")
    record("verified answers cite a page",
           answers["citation_rate"] >= THRESHOLDS["citation_rate"],
           f"citation_rate={pct(answers['citation_rate'])}")

    # ---------------- Summary ----------------
    print()
    print("=" * 66)
    print(f"SUMMARY: {passed}/{len(results)} metrics passed")
    failed = [n for n, ok in results if not ok]
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("EVALUATION: ALL METRICS PASSED")


if __name__ == "__main__":
    main()