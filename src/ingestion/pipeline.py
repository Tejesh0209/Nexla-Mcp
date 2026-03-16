import structlog
from src.ingestion.downloader import download_pdfs
from src.ingestion.parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all
from src.ingestion.embedder import embed_chunks
from src.ingestion.weaviate_store import (
    get_client,
    ensure_collection,
    is_already_indexed,
    upsert_chunks,
)

log = structlog.get_logger(__name__)


def run_ingestion_pipeline(force: bool = False) -> dict:
    # step 1 — get PDFs (local first, Drive fallback)
    pdf_paths = download_pdfs()

    # connect to weaviate and make sure the collection exists
    client = get_client()
    ensure_collection(client)

    # skip the heavy work if we already indexed everything
    if not force and is_already_indexed(client):
        log.info("already_indexed_skipping")
        client.close()
        return {"status": "skipped", "pdfs": len(pdf_paths)}

    # step 2 — parse, chunk, embed, upsert
    records  = parse_all_pdfs(pdf_paths)
    chunks   = chunk_all(records)
    embedded = embed_chunks(chunks)
    upsert_chunks(client, embedded)
    client.close()

    summary = {
        "status":   "success",
        "pdfs":     len(pdf_paths),
        "pages":    len(records),
        "chunks":   len(chunks),
        "embedded": len(embedded),
    }
    log.info("pipeline_complete", **summary)
    return summary