import os
from dataclasses import dataclass

import weaviate
import weaviate.classes as wvc
import structlog

from src.ingestion.embedder import EmbeddedChunk

log = structlog.get_logger(__name__)

COLLECTION = "DocumentChunk"


def get_client() -> weaviate.WeaviateClient:
    host = os.getenv("WEAVIATE_HOST", "localhost")
    port = int(os.getenv("WEAVIATE_PORT", "8080"))
    grpc  = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc)


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    # nothing to do if it already exists
    if client.collections.exists(COLLECTION):
        log.info("collection_exists", name=COLLECTION)
        return

    log.info("creating_collection", name=COLLECTION)

    # we bring our own vectors so vectorizer is none
    # BM25 is enabled by default on all text properties
    client.collections.create(
        name=COLLECTION,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(
                name="chunk_text",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.config.Property(
                name="chunk_id",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="doc_name",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="page_number",
                data_type=wvc.config.DataType.INT,
                skip_vectorization=True,
                index_filterable=True,
            ),
            wvc.config.Property(
                name="chunk_index",
                data_type=wvc.config.DataType.INT,
                skip_vectorization=True,
            ),
        ],
    )
    log.info("collection_created", name=COLLECTION)


def is_already_indexed(client: weaviate.WeaviateClient) -> bool:
    if not client.collections.exists(COLLECTION):
        return False
    col   = client.collections.get(COLLECTION)
    count = col.aggregate.over_all(total_count=True).total_count
    log.info("index_check", count=count)
    return count > 0


def upsert_chunks(
    client: weaviate.WeaviateClient,
    embedded: list[EmbeddedChunk],
) -> None:
    col = client.collections.get(COLLECTION)
    log.info("upserting", total=len(embedded))

    # batch insert — weaviate handles the chunking internally
    with col.batch.fixed_size(batch_size=50) as batch:
        for ec in embedded:
            batch.add_object(
                properties={
                    "chunk_text":  ec.chunk.text,
                    "chunk_id":    ec.chunk.chunk_id,
                    "doc_name":    ec.chunk.doc_name,
                    "page_number": ec.chunk.page_number,
                    "chunk_index": ec.chunk.chunk_index,
                },
                vector=ec.embedding,
            )

    log.info("upsert_done", total=len(embedded))