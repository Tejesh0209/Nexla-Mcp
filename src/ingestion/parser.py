"""
parser.py
Extracts text from PDFs page by page using PyMuPDF (fitz).
Returns structured PageRecord objects with doc_name and page_number
preserved on every record — critical for source attribution later.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import structlog

log = structlog.get_logger(__name__)

MIN_PAGE_CHARS = 30  # skip covers/blank pages below this threshold


@dataclass
class PageRecord:
    doc_name: str       # filename without extension  e.g. "P19-1598"
    page_number: int    # 1-indexed
    text: str           # raw extracted text
    char_count: int


def parse_pdf(pdf_path: Path) -> list[PageRecord]:
    """
    Extract text from every page of a single PDF.
    Skips pages with fewer than MIN_PAGE_CHARS characters.
    Returns list of PageRecord objects.
    """
    doc_name = pdf_path.stem
    records: list[PageRecord] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log.error("pdf_open_failed", path=str(pdf_path), error=str(e))
        raise

    log.info("parsing_pdf", doc=doc_name, total_pages=len(doc))

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()

        if len(text) < MIN_PAGE_CHARS:
            log.debug("skipping_sparse_page", doc=doc_name, page=i + 1)
            continue

        records.append(
            PageRecord(
                doc_name=doc_name,
                page_number=i + 1,
                text=text,
                char_count=len(text),
            )
        )

    doc.close()

    log.info(
        "pdf_parsed",
        doc=doc_name,
        pages_extracted=len(records),
        total_chars=sum(r.char_count for r in records),
    )
    return records


def parse_all_pdfs(pdf_paths: list[Path]) -> list[PageRecord]:
    """Parse all PDFs. Returns flat list of PageRecords."""
    all_records: list[PageRecord] = []
    for path in pdf_paths:
        all_records.extend(parse_pdf(path))
    log.info("all_pdfs_parsed", total_pages=len(all_records))
    return all_records