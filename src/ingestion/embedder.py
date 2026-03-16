import os
import time
from dataclasses import dataclass

import openai
import structlog

from src.ingestion.chunker import Chunk

log = structlog.get_logger(__name__)

# text-embedding-3-small is the sweet spot — cheap, fast, good enough for RAG
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # openai handles up to 2048 but 100 keeps retries cheap


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]  # 1536 dimensions


def _embed_batch(texts: list[str], client: openai.OpenAI) -> list[list[float]]:
    # retry up to 3 times — rate limits happen, especially on first run
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            return [item.embedding for item in resp.data]
        except openai.RateLimitError:
            wait = 2 * (attempt + 1)
            log.warning("rate_limited", attempt=attempt + 1, waiting=wait)
            time.sleep(wait)
        except openai.OpenAIError as e:
            log.error("embedding_failed", error=str(e))
            if attempt == 2:
                raise
            time.sleep(2)

    raise RuntimeError("embedding API gave up after 3 attempts")


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = openai.OpenAI(api_key=key)
    results: list[EmbeddedChunk] = []

    log.info("embedding_start", chunks=len(chunks), model=MODEL)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        log.info("embedding_batch", batch=i // BATCH_SIZE + 1, size=len(batch))

        vectors = _embed_batch([c.text for c in batch], client)

        for chunk, vector in zip(batch, vectors):
            results.append(EmbeddedChunk(chunk=chunk, embedding=vector))

    log.info("embedding_done", total=len(results))
    return results