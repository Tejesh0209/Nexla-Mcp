"""
chunker.py
Splits PageRecords into overlapping token-aware chunks using tiktoken.
512 tokens per chunk, 64 token overlap.
Every chunk carries doc_name + page_number for attribution.
"""

from dataclasses import dataclass

import tiktoken
import structlog

from src.ingestion.parser import PageRecord

log = structlog.get_logger(__name__)

CHUNK_TOKENS   = 512
OVERLAP_TOKENS = 64
ENCODING_NAME  = "cl100k_base"


@dataclass
class Chunk:
    chunk_id:    str   # "{doc_name}__p{page}_c{idx}"
    doc_name:    str
    page_number: int
    chunk_index: int   # 0-indexed within page
    text:        str
    token_count: int


def chunk_page(record: PageRecord, enc: tiktoken.Encoding) -> list[Chunk]:
    """Split a single PageRecord into overlapping chunks."""
    tokens  = enc.encode(record.text)
    chunks: list[Chunk] = []
    start   = 0
    idx     = 0

    while start < len(tokens):
        end         = min(start + CHUNK_TOKENS, len(tokens))
        chunk_text  = enc.decode(tokens[start:end]).strip()

        if chunk_text:
            chunks.append(Chunk(
                chunk_id    = f"{record.doc_name}__p{record.page_number}_c{idx}",
                doc_name    = record.doc_name,
                page_number = record.page_number,
                chunk_index = idx,
                text        = chunk_text,
                token_count = end - start,
            ))
            idx += 1

        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS

    return chunks


def chunk_all(records: list[PageRecord]) -> list[Chunk]:
    """Chunk all PageRecords. Returns flat list of Chunks."""
    enc        = tiktoken.get_encoding(ENCODING_NAME)
    all_chunks: list[Chunk] = []

    for record in records:
        all_chunks.extend(chunk_page(record, enc))

    log.info(
        "chunking_complete",
        total_chunks = len(all_chunks),
        avg_tokens   = round(
            sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1), 1
        ),
    )
    return all_chunks