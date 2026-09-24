"""Phase 4 - Extract text from the Hulio Product Monograph PDF.

Reads the PDF page-by-page with PyMuPDF (fitz), performs safe basic text
cleaning, and saves a structured JSON file in data/.
"""

import json
import re
import sys
from pathlib import Path

try:
    import pymupdf  # PyMuPDF (modern import)
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        sys.exit("PyMuPDF not installed. Run: pip install pymupdf")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "hulio_product_monograph.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hulio_monograph_extracted.json"

DOCUMENT_NAME = "Hulio Product Monograph"

# Footer line like "Reference ID: 5525255 4" repeats on pages and is
# regulatory/print noise, not medical content. Remove it safely.
FOOTER_PATTERN = re.compile(r"^Reference ID:\s*\d*\s*\d*$")


def clean_text(raw_text: str) -> str:
    """Remove extraction noise without changing the medical meaning."""
    if not raw_text:
        return ""

    # Replace straight/non-breaking spaces introduced during extraction.
    text = raw_text.replace("\u00a0", " ")

    # Drop control characters that are invisible or break text flow.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)

    # Remove footer noise lines.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if FOOTER_PATTERN.match(stripped):
            continue
        lines.append(stripped)

    text = "\n".join(lines)

    # Collapse runs of blank lines into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_to_pages(pdf_path: Path) -> list:
    """Return a list of {document, page, text} dicts, one per PDF page."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
        for index, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            pages.append(
                {
                    "document": DOCUMENT_NAME,
                    "page": index,
                    "text": clean_text(raw),
                }
            )
        if total_pages != len(pages):
            sys.exit("Mismatch between PDF page count and extracted pages.")
    return pages


def main() -> None:
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    pages = extract_pdf_to_pages(PDF_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)

    non_empty = sum(1 for item in pages if item["text"])
    print(f"PDF read:          {PDF_PATH.name}")
    print(f"Pages extracted:   {len(pages)}")
    print(f"Pages with text:   {non_empty} ({non_empty / max(len(pages), 1):.0%})")
    print(f"JSON written to:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()