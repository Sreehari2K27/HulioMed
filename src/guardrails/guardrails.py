"""Phase 11 - Safety layer: checks BEFORE and AFTER the RAG pipeline.

The RAG architecture from Phases 9-10 is unchanged. This module *wraps* it:

  1. INPUT CHECK   - scan the question for prompt injection, medical
                     emergencies, personal/diagnosis requests, and personal
                     information (PII). A violation blocks the question
                     BEFORE any embedding or LLM call.
  2. RAG PIPELINE  - generate_answer() (Phase 10). This already rejects
                     out-of-scope questions via the Phase 9 relevance gate,
                     so "unsupported / out-of-scope" questions get the
                     fallback message without an LLM call.
  3. OUTPUT CHECK  - verify the produced answer: picked up from the
                     monograph only (page citation present), no PII leaked,
                     non-empty.

answer_safely() returns one dict for every input, so the future API/UI has a
single, predictable contract.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import generate as gen
from src.guardrails import policy

REASON_MESSAGES = {
    "prompt_injection": policy.INJECTION_MESSAGE,
    "emergency": policy.EMERGENCY_MESSAGE,
    "personal": policy.PERSONAL_MESSAGE,
    "pii": policy.PII_MESSAGE,
}


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def check_input(question: str) -> list:
    checks = [
        {
            "check": "prompt_injection",
            "passed": not policy.is_prompt_injection(question),
        },
        {
            "check": "emergency",
            "passed": not policy.is_emergency(question),
        },
        {
            "check": "personal_request",
            "passed": not policy.is_personal_request(question),
        },
    ]
    pii = policy.find_pii(question)
    checks.append(
        {"check": "pii", "passed": not pii, "detail": ", ".join(pii) if pii else ""}
    )
    return checks


def first_blocked(checks: list):
    """Return the reason label of the first failed input check (by priority),
    or None if everything passed."""
    reason_for_check = {
        "prompt_injection": "prompt_injection",
        "emergency": "emergency",
        "personal_request": "personal",
        "pii": "pii",
    }
    ordered = ["prompt_injection", "emergency", "personal_request", "pii"]
    for check_name in ordered:
        for c in checks:
            if c["check"] == check_name and not c["passed"]:
                return reason_for_check[check_name]
    return None


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------
def check_output(response: dict) -> list:
    checks = [
        {"check": "answer_non_empty", "passed": bool(response.get("answer"))},
    ]
    if response.get("verified") is True:
        text = response.get("answer", "")
        if policy.contains_disclaimer(text):
            # Model explicitly declined: a refusal needs no citation.
            checks.append({
                "check": "grounded_with_citation",
                "passed": True,
            })
            checks.append({
                "check": "safe_explicit_refusal",
                "passed": True,
            })
        else:
            checks.append({
                "check": "grounded_with_citation",
                "passed": "page" in text.lower()
                and len(response.get("sources", [])) > 0,
            })
        checks.append({
            "check": "no_pii_in_answer",
            "passed": not policy.find_pii(text),
        })
    else:
        checks.append({
            "check": "out_of_scope_fallback",
            "passed": policy.contains_disclaimer(response.get("answer", "")),
        })
    return checks if response else []


def answer_safely(question: str) -> dict:
    """Full protected pipeline: input checks -> RAG -> output checks."""
    input_checks = check_input(question)
    reason = first_blocked(input_checks)

    if reason is not None:
        message = REASON_MESSAGES[reason]
        return {
            "question": question,
            "status": "blocked",
            "blocked_reason": reason,
            "answer": message,
            "verified": False,
            "sources": [],
            "checks": {"input": input_checks, "output": []},
        }

    response = gen.generate_answer(question)

    output_checks = check_output(response)
    status = "answered"
    if response.get("verified") is False:
        status = "answered_out_of_scope"

    return {
        "question": question,
        "status": status,
        "blocked_reason": None,
        "answer": response.get("answer", ""),
        "verified": response.get("verified", False),
        "sources": response.get("sources", []),
        "checks": {"input": input_checks, "output": output_checks},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.guardrails.guardrails "your question here"')
    import json
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(answer_safely(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))