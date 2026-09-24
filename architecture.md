# HulioMed — RAG Architecture (Phase 2)

This document describes, in very simple language, how HulioMed will work end
to end. It is a plan — no application code exists yet, no packages are
installed yet.

**Important:** This is a portfolio prototype, not a clinically validated
medical device or medical decision-support system.

---

## A. Architecture diagram (text / ASCII)

```
                         USER
                          |
                          v
              +-----------------------+
              |   Streamlit UI (13)   |
              |  chat box + prompt    |
              +-----------+-----------+
                          | question (natural language)
                          v
              +-----------------------+
              |   Query processing(7) |
              |  embed the question   |
              +-----------+-----------+
                          |
                          | question embedding
                          v
              +-----------------------+
              |   ChromaDB (6,8)      |  <--- created once in "build" step
              |  vector search        |
              |  returns top-k chunks |
              +-----------+-----------+
                          |
                          | relevant chunks (+ page numbers)
                          v
              +-----------------------+
              | Answer generation (9) |
              | LLM: chunks = context |
              +-----------+-----------+
                          |
                          | grounded answer + citations (10)
                          v
              +-----------------------+
              | Safety guardrails(11) |  no diagnosis / no advice /
              | PII protection (12)   |  disclaimers / fallback
              +-----------+-----------+
                          v
                      USER / UI
   (answer + source and page number citation)


       BUILD / OFFLINE STEP (before the user ever chats):
   +---------------+   +-----------------+   +---------------+   +-----------+
   | Hulio PDF (1) |-->| PDF text load & |-->| Clean & chunk |-->| Embed +  |
   | source doc    |   | page mapping(2)|   | (3)(4)        |   | store    |
   +---------------+   +-----------------+   +---------------+   | ChromaDB |
                                                                 +-----------+
```

There are two parts:

1. **Offline build step** — run once. Reads the PDF, cleans and chunks the
   text, creates embeddings, and stores everything in ChromaDB.
2. **Online query step** — every time the user asks a question. Embed the
   question, find matching chunks in ChromaDB, generate the grounded answer,
   add safeguards, show it with citations.

---

## B. Component-by-component explanation (very simple language)

### 1. Source document
The only knowledge source: `hulio_product_monograph.pdf`, kept unchanged in the
project. Everything the bot says must come from this file.

### 2. Document loading
A loader opens the PDF and extracts the text. PDFs store text in a compressed
way, so we decompress each page's content stream and pull out the words. As we
read each page, we remember which page every piece of text belongs to, so pages
numbers survive (page metadata is preserved).

### 3. Text cleaning
The extracted text contains noise: header/footer junk (like "Reference ID"),
stray hyphenation that splits words mid-word, odd spacing, and repeated symbols.
Cleaning fixes these so later steps produce better chunks and better answers.

### 4. Chunking
A chunk is a small, self-contained piece of text (a few sentences). The whole
monograph is too long for the model to read, so we split it into chunks of
roughly 400–600 characters with some overlap so no idea is cut in half. Each
chunk keeps metadata: **page number** and (where detectable) the **section**
(e.g., "2.1 Dosage and Administration"). These chunks become the "facts" the
bot searches.

### 5. Embeddings
**What is an embedding in plain words?** An embedding turns a piece of text
into a list of numbers (a vector) that captures its *meaning*. Words/sentences
that mean similar things get vectors that are close together.

**Why do we need it?** A user rarely types the exact words in the document
(e.g., "how much do I inject" vs. "recommended subcutaneous dosage"). Ordinary
word search fails; semantic search with embeddings succeeds because it compares
*meaning*, not matching words.

We will use the Mistral embedding model (`mistral-embed`) to convert every
chunk (offline) and every user question (online) into vectors.

### 6. Vector database
ChromaDB stores the chunks, their embeddings (vectors), and their metadata.
When it gets a question vector, it finds the chunks whose vectors are closest
(the "most similar meaning"), using a mathematical similarity measure. This is
the project's memory: it persists to disk so we only build it once.

### 7. Query processing
The user types a natural-language question. The bot converts the question into
an embedding, using the **same** embedding model as the chunks, so the question
and document live in the same "meaning space" and can be compared.

### 8. Retrieval
ChromaDB searches for the top-k most relevant chunks (we will start with a small
number, like k=4–5). Only these chunks are used as context for answering. If the
retrieved chunks are not relevant enough (below a similarity threshold), we
treat the question as "not answerable from the monograph."

### 9. Answer generation
A Mistral chat model (e.g., `open-mistral-nemo` - works on the free tier)
receives a carefully written prompt containing:
- the retrieved chunks (as the *only* context),
- instructions to answer ONLY from them, concisely and factually,
- instructions to say "could not be verified from the monograph" if the chunks
  don't contain the answer.

The model writes a short, grounded answer. It never sees any other source.

### 10. Citation
Answers include a source line, e.g., "Source: Hulio Product Monograph, Page 12."
The page number comes from the chunk metadata that was retrieved. If page
information is missing/unreliable for a section, we cite the document alone.

### 11. Safety / guardrails
The system-level rules (also coded in the prompt and validated after the
answer):
- Never diagnose any condition.
- Never give patient-specific medical advice.
- Never recommend starting, stopping, or changing treatment.
- Only make claims that appear in the retrieved monograph chunks (no invented
  or "common knowledge" additions).
- If the answer cannot be verified from retrieved monograph content, say so
  explicitly instead of guessing.

Implemented (Phase 11) as a wrapping safety layer, `src/guardrails/`, that
leaves the retrieval + generation pipeline unchanged:

- INPUT CHECKS (block the question *before* any embedding/LLM call, by
  priority): prompt injection -> medical emergency -> personal/diagnosis
  request -> personal information (PII: phone, email, PAN, Aadhaar, cards).
- RAG PIPELINE: `generate_answer()` runs unchanged; the Phase 9 relevance
  gate rejects out-of-scope / unsupported questions with a fixed fallback.
- OUTPUT CHECKS: every verified answer must carry a page citation; or be an
  explicit "could not be verified" refusal; and must not leak PII.

Each rejected question returns a fixed human-readable message with a
`blocked_reason`, giving the API/UI one predictable contract to render.

### 12. PII protection
The app must not request, process, log, or store PAN, Aadhaar, phone numbers,
emails, OTPs, account numbers, or patient identifiers. The UI will show a
notice; the backend will not persist chat history, and any non-monograph text
is treated as out-of-scope.

### 13. UI (Streamlit)
A simple chat interface with:
- A welcome message explaining the bot.
- Three example questions users can click to try.
- A persistent banner: **"Facts-only. No medical advice."**
- A chat input box.
- The answer text and a source/page citation under it.

### 14. Evaluation
A small test set of factual questions (with expected answer facts/pages), plus
"unsupported" questions (must trigger the fallback) and malicious guardrail
questions (must be refused). Implemented in Phase 14 as
`data/eval_cases.json` (labeled gold set) with metrics in
`src/evaluation/evaluate.py` and a threshold-gated report in
`src/evaluation/verify_evaluation.py`:

- Retrieval `hit@4` and relevance-gate accuracy + separation margin.
- Guardrail block-recall per category and no-over-blocking precision.
- Answer rate, fact rate (answers state the expected fact), grounding rate
  (fact appears verbatim in the retrieved context = no invention), citation
  rate (verified answers name pages).

Result (Phase 14 run): 13/13 metrics passed — hit@4 100%, gate 100%,
margin 0.131, block-recall 100% (0 misses), over-block 0, answer/fact/
grounding/citation all 100%.

---

## C. Data flow (offline build)

1. `hulio_product_monograph.pdf` → PDF loader
2. → page-by-page text (with page numbers)
3. → cleaned text
4. → chunks (with page + section metadata)
5. → Mistral embeddings (vectors)
6. → ChromaDB collection stored on disk (chunk text + vector + metadata)

Result: a ready-to-query memory of the monograph.

---

## D. Query flow (online chat)

1. User sends a question in the UI.
2. Question → embedding (same Mistral model).
3. ChromaDB similarity search → top-k chunks with metadata.
4. Relevance check: are the top chunks close enough? If not → "could not be
   verified" fallback; stop.
5. If yes: LLM prompt = chosen chunks + strict instructions.
6. LLM → concise factual answer.
7. Post-check safeguards (refuse medical-advice/PII content; verify citations).
8. UI shows answer + "Source: Hulio Product Monograph, Page X" + disclaimer.

---

## E. Technology choices and why

| Component        | Choice                  | Why it keeps things simple & beginner-friendly |
|------------------|-------------------------|------------------------------------------------|
| Language         | Python                  | One language for the whole pipeline; huge learning ecosystem |
| PDF text loading | A lightweight PDF library (e.g., `PyMuPDF`) | Extracts text per page and gives page numbers directly |
| Embeddings       | Mistral `mistral-embed` | Free/low-cost tier, one API call, easy for beginners |
| Vector DB        | ChromaDB                 | Single local store for text + vectors + metadata; tiny code footprint |
| LLM              | Mistral chat (`open-mistral-nemo`) | One prompt, one call; no framework needed |
| Backend API      | FastAPI                  | Small, clean, industry-standard REST API |
| Frontend         | Streamlit                | Chat UI in a few lines of Python; no HTML/CSS needed |
| Orchestration    | None (plain Python)      | Explicit step-by-step code teaches RAG better than LangChain |

**Deliberately NOT used:** LangChain, LangGraph, agents, multiple vector
databases, RAG frameworks, or production authentication — none are needed for a
correct, learnable prototype.

---

## F. Folder structure (for Phase 3+)

```
HulioMed/
├── hulio_product_monograph.pdf      # source document (never modified)
├── problemstatement.md              # Phase 1 - product statement
├── architecture.md                  # Phase 2 - this document
├── README.md                        # Phase 15 - portfolio docs
├── data/
│   └── chroma_db/                   # ChromaDB persisted storage
├── src/
│   ├── load_pdf.py                  # Phase 4 - extract text + page numbers
│   ├── clean_text.py                # Phase 3 (as part of loader) - cleaning
│   ├── chunking.py                  # Phase 5 - split into chunks + metadata
│   ├── embed_and_store.py           # Phase 6/7 - build ChromaDB
│   ├── retrieve.py                  # Phase 9 - query vector DB
│   ├── generate.py                  # Phase 10 - LLM answer + citations
│   ├── guardrails.py                # Phase 11 - safety + fallback checks
│   └── api.py                       # Phase 12 - FastAPI backend
├── app/
│   └── ui.py                        # Phase 13 - Streamlit chat interface
├── tests/
│   └── eval_test_set.csv            # Phase 14 - factual / unsupported / advice
└── requirements.txt                 # exact package versions
```

---

## G. Security and medical-information guardrails

- **Grounding-only answers**: the LLM sees nothing but retrieved monograph
  chunks and must not use outside knowledge.
- **Fallback when unsure**: below-threshold relevance → "could not be verified
  from the monograph."
- **Scope limits**: no diagnosis, no personal medical advice, no starting/
  stopping/changing treatment, no clinical decisions.
- **PII ban**: PAN, Aadhaar, phone, email, OTP, account numbers, patient
  identifiers are never requested or stored; no chat history persisted.
- **Transparency**: "Facts-only. No medical advice." banner + persistent
  prototype disclaimer; no implication of clinical validation.
- **No secrets in repo**: API keys go in a local `.env` file listed in
  `.gitignore`, never committed.

---

## H. Known technical limitations (architecture-level)

- PDF text extraction can garble tables or two-column layouts; page numbers are
  best-effort.
- Semantic search may miss an exact answer if wording differs a lot; chunk
  overlap mitigates but does not remove this.
- Every answer is only as good as the retrieved chunks; wrong retrieval →
  fallback or partial answer.
- An LLM can occasionally produce fluent but ungrounded text; guardrails reduce
  but cannot fully eliminate this.
- One static document only; no live safety-signal or label updates.
- English only.
- Prototype performance/cost: reasonable but not tuned for scale.

---

## I. Future improvements (not now)

- Better chunking (semantic or heading-aware splitting).
- Reranking of retrieved chunks or hybrid keyword + vector search.
- Confidence scoring displayed in the UI.
- Multi-document: adding HULIO EMA/Health Canada labels, but only if scope grows.
- Additive evaluation harness with a larger test set and golden pages.
- Optional local/on-prem embeddings (no API) to cut cost.
- Better table extraction for dosing tables.

---

*Document phase: Phase 2 of 15. Next: Phase 3 — project folder structure.*