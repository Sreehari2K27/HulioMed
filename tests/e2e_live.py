"""Phase 13 - LIVE end-to-end check: the real chat UI talking to the REAL
backend and real Mistral model.

Requires the FastAPI backend to be running. Point the UI at it:

    # terminal 1
    uvicorn app.main:app --reload

    # terminal 2 (Windows PowerShell)
    $env:HULIOMED_API_URL="http://127.0.0.1:8000"
    python tests/e2e_live.py

The UI script runs unpatched (no mocks): the question travels
UI -> HTTP /ask -> guardrails -> retrieval (ChromaDB) -> Mistral LLM
-> back into the chat bubble.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

API_URL = os.getenv("HULIOMED_API_URL", "http://127.0.0.1:8000")
CHAT_UI = str(PROJECT_ROOT / "app" / "chat_ui.py")


def all_texts(at: AppTest) -> str:
    chunks = [m.value for m in at.markdown]
    for group in ("info", "success", "warning"):
        try:
            chunks += [m.value for m in getattr(at, group)]
        except AttributeError:
            pass
    return "\n".join(chunks)


def main() -> int:
    # The app reads HULIOMED_API_URL at import time inside the AppTest session.
    os.environ["HULIOMED_API_URL"] = API_URL
    print(f"E2E UI -> {API_URL} (real backend, real Mistral LLM)")

    at = AppTest.from_file(CHAT_UI, default_timeout=90)
    at.run()

    texts = all_texts(at)
    assert "48 monograph chunks loaded" in texts, "sidebar health check missing"

    at.chat_input[0].set_value(
        "What is the recommended dose of HULIO for plaque psoriasis?"
    ).run()
    at.run()  # let the stored message render

    texts = all_texts(at)
    labels = [e.label for e in at.expander]
    assert "80 mg" in texts, "dose fact missing from the answer bubble"
    assert "Verified answer" in texts, "verified badge missing"
    assert any(label.startswith("Sources") for label in labels), "sources missing"

    print("E2E OK — chat UI answered a real question through the real backend:")
    for line in texts.splitlines():
        if "80 mg" in line:
            print("  answer preview:", line[:160])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)