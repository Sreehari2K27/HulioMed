"""Phase 10 - Answer generation: turn retrieved chunks into a grounded answer.

Flow for a user question:
  1. retrieve() (Phase 9) finds the top-k relevant monograph chunks and judges
     whether the question is on-topic (relevant=...).
  2. If NOT relevant -> return a clear "could not be verified" fallback without
     ever calling the LLM.
  3. If relevant -> call a Mistral chat model with the retrieved chunks as the
     ONLY context. The prompt tells the model to:
       - answer ONLY from those chunks,
       - never invent medication facts,
       - refuse diagnosis / patient-specific advice / treatment changes,
       - cite the page numbers when it mentions a fact.

The function returns a dict the API/UI can render directly.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve as rt  # noqa: E402

DEFAULT_LLM_MODEL = "open-mistral-nemo"
MAX_TOKENS = 350
TEMPERATURE = 0.2  # low = more factual, less "creative"

FALLBACK_MESSAGE = (
    "I could not verify this information in the Hulio Product Monograph. "
    "I can only answer based on what that document says about HULIO "
    "(adalimumab-fkjp). Please rephrase the question or ask about a topic "
    "covered in the monograph."
)


def get_llm_client():
    """Read MISTRAL_API_KEY from .env and build a Mistral client."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("MISTRAL_API_KEY missing in .env")
    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral
    return Mistral(api_key=api_key)


def build_messages(question: str, hits: list) -> list:
    """Build the system + user messages for the chat model."""
    excerpts = []
    for i, hit in enumerate(hits, start=1):
        pages = ", ".join(str(p) for p in hit["pages"])
        excerpts.append(
            f"[Excerpt {i}] from the {hit['document']}, page(s) {pages}:\n{hit['text']}"
        )
    context = "\n\n".join(excerpts)

    system = (
        "You are HulioMed, a medical-information assistant that answers "
        "questions ONLY about HULIO (adalimumab-fkjp) using excerpts from its "
        "official Hulio Product Monograph.\n"
        "Strict rules:\n"
        "- Answer ONLY from the provided excerpts. Do not use any outside "
        "medical knowledge and do not invent facts.\n"
        "- Write a concise, factual answer of at most 4 sentences.\n"
        "- When you mention a fact, cite the source page(s): "
        "'(Hulio Product Monograph, page(s) N)'.\n"
        "- If the excerpts do NOT contain an answer to the question, say "
        "exactly: 'The requested information could not be verified from the "
        "Hulio Product Monograph.' Do not guess.\n"
        "- Never diagnose a condition. Never give patient-specific medical "
        "advice. Never recommend starting, stopping, or changing any treatment. "
        "Never make treatment decisions for a patient.\n"
        "- If asked for personal advice or a diagnosis, decline and suggest "
        "consulting a qualified healthcare professional.\n"
        "- Do not answer questions unrelated to HULIO.\n\n"
        f"Here are the excerpts from the Hulio Product Monograph:\n\n{context}"
    )
    user = f"Question: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_sources(hits: list) -> list:
    """Return just the citation fields the UI needs."""
    return [
        {
            "chunk_id": h["chunk_id"],
            "document": h["document"],
            "pages": h["pages"],
            "distance": round(h["distance"], 4),
        }
        for h in hits
    ]


def chat_with_retry(client, *, model, messages, max_tokens, temperature):
    """Call the Mistral chat API, retrying on rate limits / temporary errors.

    The free Mistral tier is rate-limited, so we back off and try again.
    """
    for attempt in range(4):
        try:
            return client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            text = str(exc)
            retryable = any(
                token in text.lower()
                for token in (
                    "429",
                    "rate limit",
                    "rate_limited",
                    "timeout",
                    "temporarily unavailable",
                    "503",
                    "502",
                )
            )
            if not retryable:
                raise
            wait = 2 * (2 ** attempt)  # 2s, 4s, 8s
            print(f"  (rate limit / temporary error - retrying in {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Mistral chat API still unavailable after retries")


def generate_answer(question: str, top_k: int = rt.TOP_K) -> dict:
    """Full pipeline: retrieve context, then produce a grounded answer."""
    retrieval = rt.retrieve(question, top_k=top_k)

    # Out-of-scope question: no LLM call, clear fallback.
    if not retrieval["relevant"] or not retrieval["hits"]:
        return {
            "question": question,
            "verified": False,
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "num_sources": 0,
            "retrieval_relevant": False,
            "model": DEFAULT_LLM_MODEL,
        }

    messages = build_messages(question, retrieval["hits"])
    client = get_llm_client()
    completion = chat_with_retry(
        client,
        model=DEFAULT_LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = completion.choices[0].message.content.strip()

    return {
        "question": question,
        "verified": True,
        "answer": answer,
        "sources": render_sources(retrieval["hits"]),
        "num_sources": len(retrieval["hits"]),
        "retrieval_relevant": True,
        "model": DEFAULT_LLM_MODEL,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.generation.generate "your question here"')
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(generate_answer(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))