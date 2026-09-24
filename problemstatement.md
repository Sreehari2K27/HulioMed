# HulioMed — Product Statement (Phase 1)

## 1. Problem Statement
Medical information about HULIO (adalimumab-fkjp) is scattered across an
official regulatory document, the Hulio Product Monograph (a PDF of the U.S.
Prescribing Information). Patients, caregivers and healthcare-adjacent users
must read and search a long technical document to find answers. HulioMed is a
portfolio prototype that uses Retrieval-Augmented Generation (RAG) to answer
plain-language questions about HULIO, grounded ONLY in that monograph, so that
every answer is verifiable and clearly sourced.

## 2. Target Users
- Patients and caregivers researching HULIO (adalimumab).
- Students, interns, and healthcare-adjacent professionals practicing
  medical-information retrieval.
- Developers / recruiters reviewing a RAG portfolio project.

## 3. User Need
Users want quick, factual answers about indications, dosing, storage, and
safety of HULIO, without reading a 100+ page PDF, and without being misled by
unverified or invented information from general web sources.

## 4. Product Goal
Build a beginner-friendly RAG chatbot that answers questions about HULIO using
the Hulio Product Monograph as the single knowledge source, returns the
relevant source and page number with each answer, and refuses to guess or to
give medical advice.

## 5. In Scope
- Phase 1: This document (product statement).
- Reading and chunking the Hulio Product Monograph PDF.
- Embedding chunks (Mistral embeddings) and storing them in ChromaDB.
- Retrieval of relevant chunks by semantic similarity.
- Answer generation grounded strictly in retrieved chunks (Mistral LLM).
- A confidence-based "not found in the monograph" fallback response.
- Safety guardrails (no diagnosis, no personal advice, no PII).
- Simple FastAPI backend and Streamlit chat UI.
- Basic evaluation and a portfolio README.

## 6. Out of Scope
- Any medical database other than the monograph (no external medical sources,
  websites, or third-party references).
- Clinical validation, regulatory approval, or use in real patient care.
- Replacing professional medical judgment.
- Multi-document retrieval, agents, or LangChain-style orchestration.
- PII capture or storage of any kind.
- Mobile apps, production deployment, or authentication/authorization.
- Logging or storing chat history containing personal data.

## 7. Functional Requirements
- User can ask a natural-language question about HULIO.
- System retrieves the most relevant chunks from the monograph.
- System answers ONLY from retrieved chunks; each factual answer cites the
  source document (and page number where available).
- If nothing relevant or sufficient is found, the system clearly states the
  information could not be verified from the monograph.
- System gives a snapshot citation like "Hulio Product Monograph, Page N".
- Key facts (indications, dosing) must match the monograph (e.g., RA 40 mg
  every other week; Crohn's 160 mg Day 1 / 80 mg Day 15 / 40 mg every other
  week from Day 29).

## 8. Non-Functional Requirements
- Simplicity: plain Python + Mistral + ChromaDB + FastAPI + Streamlit. No
  unnecessary frameworks.
- Performance: typical question answered in a few seconds; retrieves from a
  single in-project vector DB.
- Grounding quality: generated text must be traceable to retrieved chunks.
- Maintainability: small, commented, step-by-step modules that a beginner can
  follow.
- Cost-conscious: minimal embedding + LLM calls for a prototype.

## 9. Medical Information Safety Guardrails
- No diagnosis: HulioMed never diagnoses any condition.
- No patient-specific medical advice: never recommends starting, stopping,
  dosing, or changing any treatment.
- No clinical decisions: the bot informs, it does not decide.
- Grounding-only: answers must come from retrieved monograph content; the model
  must not add outside knowledge.
- When unable to verify from the monograph, respond exactly that (do not
  speculate).
- Disclaimers: a persistent prototype notice that this is for learning and must
  not replace a licensed healthcare professional.

## 10. PII Protection
- HulioMed must not request, process, log, or store PII: name, PAN, Aadhaar,
  phone number, email, OTP, account numbers, or patient identifiers.
- The chat system must not ask for or remember personal health data.
- Retrieval and storage contain only monograph text, never user identity.

## 11. Success Criteria
- 80%+ of a set of monograph-answerable sample questions return correct,
  sourced answers.
- Questions outside the monograph get the explicit "could not be verified"
  fallback instead of a guess.
- Every generated answer includes at least one source reference (document /
  page).
- The bot refuses (does not answer) diagnosis and personal-advice questions.
- The full pipeline runs locally from README instructions: load → chunk →
  embed → retrieve → answer.

## 12. Known Limitations
- Prototype, not a clinical or regulatory product.
- Text extraction from PDF may lose table structure or page-perfect layout; page
  numbers are best-effort.
- Semantic retrieval can miss exact-phrase answers if the monograph uses
  different wording.
- LLM may produce fluent but ungrounded text; guardrails reduce, not eliminate,
  this risk.
- Only one static document is used; no real-time or updated-safety-signal
  coverage.
- No handling of multiple/hypothetical patient scenarios beyond the monograph.
- English only.

---

*Document phase: Phase 1 of 15. Next: Phase 2 — RAG architecture.*