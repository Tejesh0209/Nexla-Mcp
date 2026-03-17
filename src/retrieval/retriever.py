import os
from dataclasses import dataclass

import weaviate
import weaviate.classes as wvc
from openai import OpenAI
import structlog

log = structlog.get_logger(__name__)

COLLECTION   = "DocumentChunk"
DEFAULT_TOP_K = 6
ALPHA         = 0.75  # 0 = pure BM25, 1 = pure vector, 0.75 leans semantic


@dataclass
class RetrievedChunk:
    chunk_id:    str
    doc_name:    str
    page_number: int
    text:        str
    score:       float  # hybrid fusion score from Weaviate


def _embed_query(question: str) -> list[float]:
    # same model we used at index time — must match or scores are meaningless
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp   = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    return resp.data[0].embedding


def hybrid_search(
    client: weaviate.WeaviateClient,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Run a hybrid search over the DocumentChunk collection.
    BM25 handles exact keyword matches, vector handles semantic similarity.
    alpha=0.75 weights results toward semantic — good default for Q&A.
    Returns top_k chunks ranked by the fused hybrid score.
    """
    log.info("hybrid_search", question=question[:80], top_k=top_k, alpha=ALPHA)

    query_vector = _embed_query(question)
    collection   = client.collections.get(COLLECTION)

    results = collection.query.hybrid(
        query=question,           # BM25 uses this
        vector=query_vector,      # dense search uses this
        alpha=ALPHA,
        limit=top_k,
        return_metadata=wvc.query.MetadataQuery(score=True),
        return_properties=["chunk_text", "chunk_id", "doc_name", "page_number"],
    )

    chunks = []
    for obj in results.objects:
        p = obj.properties
        chunks.append(RetrievedChunk(
            chunk_id    = p["chunk_id"],
            doc_name    = p["doc_name"],
            page_number = int(p["page_number"]),
            text        = p["chunk_text"],
            score       = obj.metadata.score or 0.0,
        ))

    log.info(
        "retrieval_done",
        returned=len(chunks),
        top_score=round(chunks[0].score, 4) if chunks else 0,
        docs_hit=list({c.doc_name for c in chunks}),
    )
    return chunks