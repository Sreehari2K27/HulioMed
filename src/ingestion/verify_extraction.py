"""Phase 4 - Verify the PDF extraction output.

Checks that the PDF opens, every page has a number, most pages have text, and
the JSON file was created correctly.
"""

import json
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:
    sys.exit("PyMuPDF not installed. Run: pip install pymupdf")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "hulio_product_monograph.pdf"
JSON_PATH = PROJECT_ROOT / "data" / "hulio_monograph_extracted.json"

MIN_TEXT_RATIO = 0.95  # at least 95% of pages should have text


def check(ok: bool, message: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {message}")
    return ok


def main() -> None:
    all_passed = True

    # 1. The PDF can be opened.
    try:
        with pymupdf.open(PDF_PATH) as doc:
            pdf_pages = doc.page_count
    except FileNotFoundError:
        all_passed = check(False, f"PDF not found at {PDF_PATH}") and all_passed
        sys.exit(1)
    except Exception as exc:
        all_passed = check(False, f"PDF could not be opened: {exc}") and all_passed
        sys.exit(1)
    all_passed = check(pdf_pages > 0, f"PDF opens with {pdf_pages} pages") and all_passed

    # 2. The JSON output exists.
    if not JSON_PATH.exists():
        check(False, f"JSON output not found at {JSON_PATH}")
        sys.exit(1)
    all_passed = check(True, "JSON output file exists") and all_passed

    # 3. JSON structure checks.
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)
    all_passed = check(isinstance(records, list) and len(records) > 0,
                       f"JSON contains a non-empty list of {len(records)} records") and all_passed

    required_keys = {"document", "page", "text"}
    all_passed = check(all(required_keys <= set(r.keys()) for r in records),
                       "every record has document, page and text") and all_passed

    # 4. Page numbers are present, complete and unique.
    page_numbers = [r["page"] for r in records]
    all_passed = check(len(page_numbers) == len(set(page_numbers)),
                       "page numbers are unique") and all_passed
    all_passed = check(page_numbers == list(range(1, len(page_numbers) + 1)),
                       "page numbers are a complete 1..N sequence") and all_passed
    all_passed = check(page_numbers == list(range(1, pdf_pages + 1)),
                       "extracted page count matches PDF page count") and all_passed

    # 5. Most pages contain text.
    non_empty = [r for r in records if r["text"]]
    ratio = len(non_empty) / len(records)
    all_passed = check(ratio >= MIN_TEXT_RATIO,
                       f"text present on {len(non_empty)}/{len(records)} pages ({ratio:.0%})") and all_passed

    # 6. Document name is correct on all records.
    all_passed = check(all(r["document"] == "Hulio Product Monograph" for r in records),
                       "document name is 'Hulio Product Monograph' on all records") and all_passed

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()