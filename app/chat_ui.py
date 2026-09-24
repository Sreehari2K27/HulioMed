"""Phase 13 - HulioMed chat UI (Streamlit).

Runs on TOP of the Phase 12 FastAPI backend. Every question you type is sent
to POST /ask; the answer bubble shows whether it was a verified, monograph-
grounded answer, an out-of-scope fallback, or a blocked request (guardrail).

Run the backend first (in another terminal, from the project root):
    uvicorn app.main:app --reload

Then start this UI:
    streamlit run app/chat_ui.py

No chat history is ever written to disk (PII safety requirement); the history
lives only in memory for the current browser session.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Make the project root importable no matter where Streamlit starts this script
# (e.g. Streamlit Community Cloud runs from the repo root). Without this, the
# `app` package cannot be found even though chat_ui.py lives inside it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import api_client

DEFAULT_LOCAL_API_URL = "http://127.0.0.1:8000"


def _resolve_default_api_url() -> str:
    """Backend address: environment variable, then Streamlit secrets, then local."""
    url = os.getenv("HULIOMED_API_URL")
    if url:
        return url
    try:
        url = st.secrets.get("HULIOMED_API_URL") or None
    except Exception:  # noqa: BLE001 - secrets may be unavailable at import
        url = None
    return url or DEFAULT_LOCAL_API_URL


DEFAULT_API_URL = _resolve_default_api_url()
APP_TITLE = "HulioMed"
APP_SUBTITLE = (
    "Medical information assistant for **HULIO (adalimumab-fkjp)**, grounded "
    "only in the Hulio Product Monograph."
)

EXAMPLE_QUESTIONS = [
    "What is the recommended dose of HULIO for plaque psoriasis?",
    "How should the HULIO prefilled pen be stored?",
    "Can HULIO be used during pregnancy?",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    base_url = None
    with st.sidebar:
        st.header(APP_TITLE)
        st.caption(APP_SUBTITLE)

        st.subheader("Backend")
        base_url = st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
            help="Address of the Phase 12 FastAPI backend.",
        )
        status = api_client.health(base_url)
        if status is None:
            st.warning(
                "Cannot reach the API. Start it with:\n"
                "`uvicorn app.main:app --reload`"
            )
        else:
            st.success(
                f"API ok — {status.get('chunks')} monograph chunks loaded, "
                f"model: {status.get('llm_model')}."
            )

        st.subheader("Try asking")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

        st.subheader("Safety & privacy")
        st.caption(
            "Educational prototype only — not medical advice. The assistant "
            "refuses diagnosis, personal treatment advice, and questions that "
            "are outside the monograph. No question or answer is stored on "
            "disk."
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return base_url


# ---------------------------------------------------------------------------
# Answer rendering
# ---------------------------------------------------------------------------
def build_assistant_markdown(result: dict) -> str:
    """Turn an API response into the markdown shown in the chat bubble."""
    status = result.get("status")
    answer = result.get("answer", "")
    if status == "blocked":
        reason = result.get("blocked_reason", "request")
        return f":red[**Question blocked**] — reason: `{reason}`\n\n{answer}"
    if status == "answered_out_of_scope":
        return f":orange[**No in-scope monograph content.**]\n\n{answer}"
    verified = (
        ":green[**Verified answer** — sourced from the Hulio Product "
        "Monograph with page citations.]"
    )
    return f"{answer}\n\n{verified}"


def render_evidence(result: dict) -> None:
    """Sources + safety checks, shown under the answer in expanders."""
    sources = result.get("sources") or []
    if sources:
        lines = []
        for s in sources:
            pages = ", ".join(str(p) for p in s["pages"])
            lines.append(
                f"- **page(s) {pages}** — {s['document']}\n"
                f"  <small>`{s['chunk_id']}` · distance {s['distance']:.3f}</small>"
            )
        with st.expander(f"Sources ({len(sources)})", expanded=True):
            st.markdown("Answer grounded in these monograph passages.")
            st.markdown("\n".join(lines), unsafe_allow_html=True)

    checks = result.get("checks") or {}
    blocks = []
    for group in ("input", "output"):
        for check in checks.get(group, []):
            icon = ":green[✓]" if check.get("passed") else ":red[✗]"
            blocks.append(f"{icon} `{check['check']}`")
    if blocks:
        with st.expander("Safety checks"):
            st.markdown("\n".join(blocks))


def render_assistant_bubble(result: dict) -> None:
    with st.chat_message("assistant"):
        st.markdown(build_assistant_markdown(result))
        if result.get("sources"):
            render_evidence(result)


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

BASE_URL = render_sidebar()

st.markdown(f"# {APP_TITLE}")
st.caption(APP_SUBTITLE)
st.markdown(
    "<span style='background-color:#E0ECEB;color:#0E7C7B;border-radius:12px;"
    "padding:2px 10px;font-size:0.8rem;font-weight:600;'>Educational "
    "prototype — not medical advice</span>",
    unsafe_allow_html=True,
)
st.info(
    "**Grounded answers.** Every answer is sourced from the Hulio Product "
    "Monograph with page citations. This assistant refuses diagnosis, "
    "personal treatment advice, and out-of-scope topics. Nothing you type "
    "is stored."
)

for message in st.session_state.messages:
    avatar = "🩺" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_evidence(message.get("result") or {})


def handle_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching the Hulio Product Monograph…"):
        result = api_client.ask(question, BASE_URL)
    if result is None:
        content = (
            f":red[**Could not reach the HulioMed API** at `{BASE_URL}`.]\n\n"
            "- **Running locally?** Start the backend first, in a terminal from "
            "the project root:\n"
            "  ```\nuvicorn app.main:app --reload\n  ```\n"
            "- **Running on the cloud?** Make sure the `HULIOMED_API_URL` "
            "secret points at the deployed backend, e.g. "
            "`https://hulio-med-api.onrender.com`.\n"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "result": None}
        )
        st.rerun()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": build_assistant_markdown(result),
            "result": result,
        }
    )
    st.rerun()


question = st.chat_input("Ask about HULIO (adalimumab-fkjp)...")
if question:
    handle_question(question)

pending = st.session_state.pop("pending_question", None)
if pending:
    handle_question(pending)

if not st.session_state.messages:
    st.markdown(
        "**Ask HulioMed anything about HULIO (adalimumab-fkjp)** — try one of "
        "the examples below, or type your own question. Every answer is "
        "grounded in the monograph and shown with its page citations."
    )
    chips = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(chips, EXAMPLE_QUESTIONS):
        with col:
            label = " ".join(question.split()[:5]) + "…"
            if st.button(
                label,
                key=f"example_{question[:18]}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.rerun()

st.markdown("---")
st.caption(
    "Educational prototype. Not medical advice. Always consult a healthcare "
    "professional for personal medical questions."
)