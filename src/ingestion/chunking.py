"""Phase 5 - Split the extracted monograph text into searchable chunks.

Reads data/hulio_monograph_extracted.json, divides the text into chunks of
roughly 500-800 words with a small overlap, and keeps the original wording
exactly as extracted. Saves the result to data/hulio_chunks.json.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "hulio_monograph_extracted.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hulio_chunks.json"

DOCUMENT_NAME = "Hulio Product Monograph"

WINDOW_WORDS = 700    # upper limit for words in one chunk
OVERLAP_WORDS = 60    # words shared between consecutive chunks
MIN_CUT_WORDS = 450   # do not cut a sentence boundary before this many words

SENTENCE_END_CHARS = (".", "?", "!", ":")


def is_sentence_end(token: str) -> bool:
    """True if a word/token likely ends a sentence."""
    return token.endswith(SENTENCE_END_CHARS)


def build_token_list(records: list) -> list:
    """Flatten every page's text into ordered tokens, each tagged by page."""
    tokens = []
    for record in records:
        for word in record["text"].split():
            tokens.append({"word": word, "page": record["page"]})
    return tokens


def build_chunks(tokens: list) -> list:
    """Slide a word window across the ordered tokens to create chunks."""
    chunks = []
    index = 0
    total = len(tokens)

    while index < total:
        remaining = total - index

        # Small leftover at the very end: use it as one final chunk and stop.
        if remaining <= MIN_CUT_WORDS:
            cut = total
            emit_chunk(chunks, tokens, index, cut)
            break

        window_end = min(index + WINDOW_WORDS, total)

        # Prefer ending the chunk after the last sentence boundary that is at
        # least MIN_CUT_WORDS deep into the chunk; otherwise use the window end.
        cut = window_end
        for pos in range(window_end - 1, index + MIN_CUT_WORDS - 1, -1):
            if is_sentence_end(tokens[pos]["word"]):
                cut = pos + 1
                break

        next_start = emit_chunk(chunks, tokens, index, cut)

        if next_start <= index:
            next_start = index + 1  # safety: always move forward
        index = next_start

    return chunks


def emit_chunk(chunks: list, tokens: list, start: int, end: int) -> int:
    """Create one chunk from tokens[start:end] and return the next start index."""
    piece = tokens[start:end]
    pages = sorted({item["page"] for item in piece})
    chunks.append(
        {
            "chunk_id": f"hulio_{len(chunks) + 1:04d}",
            "document": DOCUMENT_NAME,
            "pages": pages,
            "text": " ".join(item["word"] for item in piece),
        }
    )
    return end - OVERLAP_WORDS


def report(chunks: list) -> None:
    """Print a short summary of what was created."""
    word_counts = [len(chunk["text"].split()) for chunk in chunks]
    multi_page = sum(1 for chunk in chunks if len(chunk["pages"]) > 1)
    avg = sum(word_counts) / len(word_counts)
    print(f"Chunks created:       {len(chunks)}")
    print(f"Words (avg / min / max): {avg:.0f} / {min(word_counts)} / {max(word_counts)}")
    print(f"Overlap used:         {OVERLAP_WORDS} words")
    print(f"Chunks spanning pages: {multi_page}")
    print(f"Chunking window:      {WINDOW_WORDS} words (max), sentence-aware cut >= {MIN_CUT_WORDS}")
    print(f"JSON written to:      {OUTPUT_PATH}")


def main() -> None:
    if not INPUT_PATH.exists():
        sys.exit(f"Extracted JSON not found: {INPUT_PATH}")

    with open(INPUT_PATH, encoding="utf-8") as fh:
        records = json.load(fh)

    tokens = build_token_list(records)
    chunks = build_chunks(tokens)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    report(chunks)


if __name__ == "__main__":
    main()