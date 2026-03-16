import structlog
import logging
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
)

from src.ingestion.downloader import download_pdfs
from src.ingestion.parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all
from src.ingestion.embedder import embed_chunks
from src.ingestion.weaviate_store import (
    get_client, ensure_collection,
    is_already_indexed, upsert_chunks
)

pdfs     = download_pdfs()
records  = parse_all_pdfs(pdfs)
chunks   = chunk_all(records)
embedded = embed_chunks(chunks)

client = get_client()
ensure_collection(client)

if is_already_indexed(client):
    print("\n✓ Weaviate already has data — skipping upsert")
else:
    upsert_chunks(client, embedded)
    print("\n✓ Upserted to Weaviate")

col   = client.collections.get("DocumentChunk")
count = col.aggregate.over_all(total_count=True).total_count
print(f"✓ Weaviate has {count} chunks indexed")
client.close()