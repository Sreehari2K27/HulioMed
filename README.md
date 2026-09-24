# HulioMed

**Grounding a medical-information chat assistant at every layer: an AI Product
Management case study.**

HulioMed is an educational AI medical-information prototype that answers
questions about **HULIO (adalimumab-fkjp)** — a biosimilar of adalimumab — using
**only** the official Hulio Product Monograph (81 pages) as its knowledge
source. Every answer is generated from retrieved monograph passages and shown
with its page citations; the system refuses to guess, refuses personal medical
advice, and clearly marks when a question is out of scope.

> **Important:** This is an educational prototype for learning and portfolio
> demonstration. It is **not a clinically validated medical device**, not a
> diagnostic tool, and never a substitute for professional medical judgment.

---

## 1. Product overview
- A chat assistant ("Medical Information RAG chatbot") grounded exclusively in
  one regulatory source document — the *Hulio Product Monograph*.
- Gives complete, glanceable answers with **page-level citations** so every
  claim is traceable.
- **Evaluated, not just demonstrated**: a labeled gold set, fixed acceptance
  thresholds, and a reproducible metric report.
- Fully deployed: FastAPI backend on Render, Streamlit frontend on Streamlit
  Community Cloud, Mistral models via API keys held only in platform secrets.

## 2. Problem statement
General-purpose LLMs are fluent but **ungrounded**: they can invent dosage
values, merge drug data, and give no citable source. In medical-information
requests that is not a quality problem — it is a safety problem. The full
answer text exists in a dense 81-page regulatory label, which is hard for
people to search quickly and exactly. The problem: how to get fast, *trusted*
answers about a drug, where the answer is provably drawn from the approved
label, with the source shown.

## 3. Target users
*(Illustrative; no formal user research was conducted for this prototype.)*
- **Healthcare professionals / call-center med-info staff** who need quick,
  label-traceable answers ("What is the dose for plaque psoriasis?").
- **Product managers & engineering reviewers** evaluating a grounded-RAG
  design: retrieval quality, safety behavior, and evaluation methodology.
- **Learners** exploring how embeddings, vector search, generation and
  guardrails fit together in one working system.

## 4. Product goal
A chat assistant whose answers are (a) useful at a glance, (b) **provably
grounded** in the monograph with page citations, and (c) **safe by default** —
refusing diagnosis, personal treatment advice, emergencies, injected
prompts, out-of-scope topics, and personal data. Non-goals by design: no
multi-drug comparisons, no patient-specific reasoning, no chat-history
persistence, no PHI, no production auth/HTTPS.

## 5. Key user experience / features
- **Chat UI** with a welcomed empty state and clickable example questions.
- **Color-coded answers**: green *Verified answer* with **Sources**
  expander (page numbers first), amber *No in-scope monograph content*
  fallback, red *Question blocked — reason* for guardrail refusals.
- **Safety checks** expander per answer showing every input/output guardrail
  that ran.
- **Waiting indicator** while the monograph is searched, and friendly guidance
  when the backend is unreachable.
- **Privacy**: nothing is stored; chat history lives only in browser memory;
  personal data is refused.

## 6. How the RAG system works
1. **Extract** — PyMuPDF reads the 81-page monograph, keeping page numbers.
2. **Chunk** — ~700-word windows, 60-word overlap, sentence-aware; 48 chunks,
   each tagged with its page range.
3. **Embed** — Mistral `mistral-embed` turns each chunk into a 1024-dimension
   vector (meaning = closeness in vector space).
4. **Store & retrieve** — ChromaDB (cosine) holds 48 vectors; a user question
   is embedded and the **top-4** closest chunks are returned.
5. **Relevance gate** — if the closest chunk is farther than distance **0.30**,
   the question is out of scope and the bot answers *"could not be verified"*
   **without calling the LLM**.
6. **Generate** — Mistral `open-mistral-nemo` (temperature 0.2) answers using
   the returned chunks as its *only* context, and must either answer from them
   and cite pages, or say it cannot verify.
7. **Check output** — the answer must carry a citation (or an explicit
   refusal) and must not leak personal data.

## 7. Architecture

```
 hulio_product_monograph.pdf (81 pages)
   │  extract → chunk → embed (offline build)
   ▼
 ChromaDB (48 vectors, cosine, collection "hulio_monograph")
   ▲
 question → INPUT GUARDRAILS (injection / emergency / personal / PII)
   │  clean
   ▼
 embed → top-4 chunks → relevance gate (≤ 0.30)
   │  on-topic                │ off-topic
   ▼                          ▼
 Mistral generation      "could not be verified"
   │  + page citations        (no LLM call)
   ▼
 OUTPUT CHECKS (citation present / no PII)
   ▼
 FastAPI /ask ◄── Streamlit chat UI (sources + safety shown)
```

Both the offline build and the online query path live in this repo, with a
verification script at each layer.

## 8. Guardrails and safety considerations
- **Input guardrails run first**, before any embedding/LLM spend, in priority
  order: **prompt-injection → emergency → personal → PII**. Each maps to a
  plain-language refusal.
- Safety measured on the labeled set: **100% block recall** over 13 must-block
  cases (injection ×4, personal ×4, emergency ×2, PII ×3), **zero wrong
  blocks** of benign questions.
- Guardrails **reduce risk, they do not eliminate it** (see §14).

## 9. Source citation / grounding approach
- The model is prompted to answer **only** from the retrieved chunks and to
  state page references; an output check requires a citation **or** an explicit
  "could not be verified" refusal.
- Grounding is **provable**: in evaluation, 100% of the expected facts appear
  verbatim in the retrieved context (no invented facts), and 100% of verified
  answers carried citations.
- Citations render page-first in the UI's Sources expander.

## 10. Evaluation / testing approach
**Method:** a labeled gold set (`data/eval_cases.json`, 34 cases: 11 retrieval,
17 guardrails, 6 answers) is run through the unchanged pipeline; metrics are
computed by `src/evaluation/evaluate.py` and gated by fixed thresholds in
`src/evaluation/verify_evaluation.py`.

| Metric | What it measures | Threshold | Result |
|---|---|---|---|
| Retrieval hit@4 | top-4 chunks contain the true pages | ≥ 85% | **100%** (8/8) |
| Relevance gate accuracy | relevance decision matches label | ≥ 95% | **100%** (11/11) |
| Separation margin | closest off-topic − farthest on-topic distance | ≥ 0.10 | **0.131** |
| Guardrail block recall | must-block questions blocked correctly | ≥ 95% | **100%** (13/13, 0 misses) |
| No over-blocking | benign questions wrongly blocked | = 0 | **0** |
| Answer rate | factual questions answered + verified | ≥ 80% | **100%** (6/6) |
| Fact rate | answers contain the expected fact | ≥ 80% | **100%** |
| Grounding rate | fact appears verbatim in retrieved context | ≥ 90% | **100%** |
| Citation rate | verified answers name a page | ≥ 95% | **100%** |

The margin (0.131) justifies the 0.30 threshold: on-topic queries cluster at
0.138–0.193, off-topic at 0.315–0.356.

**Layered verification:** 10 suites, **134 checks, all green** (extraction 9,
chunking 9, store incl. similarity, database 17, retrieval 13, generation 26,
guardrails 23, evaluation 13, API 14, UI 5) plus a live end-to-end test that
pushes a question from the real UI through the real backend, vector store, and
LLM. All suites rerun in one command each.

## 11. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Python 3.12 | Language | Broad RAG ecosystem; readable for learners |
| PyMuPDF | PDF text | Lightweight, per-page control |
| Custom chunker | ~700 words, 60-word overlap | Free of heavy frameworks; one topic per chunk |
| Mistral `mistral-embed` | Embeddings (1024-d) | Free-tier, coherent vendor with the LLM |
| ChromaDB | Vector store (cosine) | Local, no server; ideal for portfolio/demo |
| Mistral `open-mistral-nemo` | Generation LLM | Free-tier friendly, strong grounded Q&A |
| FastAPI + Pydantic | API contract | Typed `/health` + `/ask`, auto `/docs` |
| Streamlit | Chat UI | Fast path to a testable chat app |
| pytest (+ AppTest, TestClient) | Testing | Scriptable, CI-friendly suite per layer |
| No LangChain/agent frameworks | Pipeline | Each step small, verifiable, explainable |

## 12. Deployment architecture
- **Frontend:** Streamlit Community Cloud — reads backend URL from the platform
  secret `HULIOMED_API_URL` (never hardcoded).
- **Backend:** FastAPI in a Docker container on Render; the container
  (re)builds the 48-vector ChromaDB store at start from the committed
  embeddings file (no API key needed for that step).
- **Secrets:** `MISTRAL_API_KEY` exists only as a Render secret; `EMBEDDING_MODEL`
  / `LLM_MODEL` come from env vars; `.env` and the local vector store are
  gitignored. CORS defaults to allow-all for the demo and is env-configurable.
- **URLs:** backend `https://hulio-med-api.onrender.com` (`/health` live,
  `chunks: 48`); frontend on Streamlit (see §16).

## 13. Key product decisions and trade-offs
- **Single source of truth.** Embedding only the monograph makes "grounded" a
  *provable* property (100% grounding rate) rather than a prompt hope.
  Trade-off: narrow scope — topics outside the monograph are refused.
- **Relevance gate instead of always-answering.** Off-topic questions stop at
  retrieval with an explicit fallback and no LLM spend. Trade-off: a mis-tuned
  threshold could block borderline questions; hence the margin metric.
- **Block before spend.** Guardrail refusals run before any API call — safer
  and cheaper. Trade-off: regex detection may miss novel phrasing.
- **Deliberate model switch.** `mistral-small-latest` rate-limited on the free
  tier, so generation uses `open-mistral-nemo` — a documented accuracy-vs-cost
  decision.
- **Citation as a contract.** The prompt and the output check both enforce
  "cite or refuse", so traceability is enforced, not hoped for.
- **Small labeled evaluation with thresholds.** Quality is repeatable and
  explainable. Trade-off: it is a small prototype sample, not a clinical study.
- **No framework, no persistence.** Simplicity and privacy (PII-safe); the
  trade-off is implementation effort for features like memory and auth.

## 14. Limitations
- **Educational prototype**, not a medical device; not clinically validated.
- **Single-document scope** — answers only from the monograph; any other
  content is out of scope.
- **Prototype-calibrated** relevance threshold and evaluation set (34 cases),
  not production-grade calibration.
- **Regex-based input guardrails** may miss novel adversarial phrasing; LLM
  output remains nondeterministic and can still be fluent but wrong.
- **No user research, no real business metrics** — target users are
  illustrative and all "results" are product/engineering measurements, not
  clinical or business outcomes.
- **Free-tier behavior:** model rate limits and Render sleep/cold-start delays
  mean occasional slow responses.
- **No authentication, PHI, or multi-user persistence** — deliberately out of
  scope for a public demo.

## 15. Future improvements
(No commitments — candidate directions only.)
- Larger, category-balanced evaluation sets and per-question human review.
- Hybrid retrieval (keyword + vector) and re-ranking for edge cases.
- Confidence scoring / uncertainty signaling on low-margin retrievals.
- Session-based disambiguation with an (explicit) "for education only" banner.
- Stricter, configurable CORS and rate limiting; optional auth for non-public
  use.
- Another grounded source document (e.g., a second monograph) to demonstrate
  multi-source attribution.

## 16. Live demo
- **Streamlit UI (live):** https://huliomed.streamlit.app
- **Backend API:** https://hulio-med-api.onrender.com — `GET /health` →
  `{"status":"ok","chunks":48}`; interactive docs at `/docs`.
- Try in the demo: *"What is the recommended dose of HULIO for plaque
  psoriasis?"* (verified answer with citations) · *"What is the weather in
  Mumbai today?"* (out-of-scope fallback) · *"My father has joint pain —
  should he start HULIO?"* (blocked: personal).

## 17. GitHub repository
- **Source:** https://github.com/Sreehari2K27/HulioMed
- Contains the full pipeline, all verification suites, the evaluation gold
  set, deployment files (`Dockerfile`, `render.yaml`), and this documentation.

---

*Portfolio case study by the project author. This document accompanies a fully
working, deployed prototype; all metrics above come from the in-repo
verification and evaluation suites, not from user or clinical studies.*