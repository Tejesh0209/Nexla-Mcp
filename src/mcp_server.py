import os
import time
import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.observability.logging_config import setup_logging
from src.observability.metrics import metrics
from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.weaviate_store import get_client, COLLECTION
from src.retrieval.router import classify
from src.retrieval.retriever import hybrid_search
from src.synthesis.synthesizer import synthesize

import structlog
log = structlog.get_logger(__name__)

setup_logging()

# ── Pydantic models for input validation ──────────────────────────

class QueryInput(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          description="Natural language question about the documents")
    strategy: str = Field(default="auto",
                          description="Override routing: 'auto', 'rag', or 'hybrid'")
    top_k: int    = Field(default=5, ge=1, le=20,
                          description="Number of chunks to retrieve")


class DocInfoInput(BaseModel):
    doc_name: str = Field(..., min_length=1,
                          description="Document name without extension e.g. 'P19-1598'")


# ── FastMCP server ────────────────────────────────────────────────

mcp = FastMCP(
    name="nexla-document-qa",
    version="1.0.0",
    description="MCP server for Q&A over academic PDF documents using hybrid RAG",
)

# shared Weaviate client — opened once at startup, closed on shutdown
_weaviate_client = None


def get_weaviate():
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = get_client()
    return _weaviate_client


# ── Tool 1: query_documents ───────────────────────────────────────

@mcp.tool()
def query_documents(question: str, strategy: str = "auto", top_k: int = 5) -> dict:
    """
    Ask a natural language question about the indexed PDF documents.
    Returns a grounded answer with source attribution (document name and page number).

    Args:
        question: Your question about the documents
        strategy: 'auto' lets the router decide, 'rag' forces Mixtral,
                  'hybrid' forces DeepSeek for cross-document reasoning
        top_k:    Number of chunks to retrieve (default 5, max 20)
    """
    # validate inputs with pydantic
    inp = QueryInput(question=question, strategy=strategy, top_k=top_k)

    start = time.time()

    # route — auto means let the classifier decide
    if inp.strategy == "auto":
        route = classify(inp.question)
    else:
        route = inp.strategy
        log.info("strategy_overridden", strategy=route)

    # retrieve
    client = get_weaviate()
    chunks = hybrid_search(client, inp.question, top_k=inp.top_k)

    if not chunks:
        return {
            "answer":       "No relevant content found in the indexed documents.",
            "sources":      [],
            "strategy":     route,
            "model":        "none",
            "tokens_used":  0,
        }

    # synthesize
    answer = synthesize(inp.question, chunks, route)

    # record metrics
    elapsed = round((time.time() - start) * 1000, 1)
    metrics.record(route, answer.tokens_used, elapsed)

    log.info(
        "query_complete",
        strategy     = route,
        model        = answer.model,
        tokens       = answer.tokens_used,
        latency_ms   = elapsed,
        sources      = [f"{s['doc_name']} p{s['page_number']}" for s in answer.sources[:3]],
    )

    return {
        "answer":      answer.text,
        "sources":     answer.sources,
        "strategy":    answer.strategy,
        "model":       answer.model,
        "tokens_used": answer.tokens_used,
        "latency_ms":  elapsed,
    }


# ── Tool 2: list_documents ────────────────────────────────────────

@mcp.tool()
def list_documents() -> dict:
    """
    List all documents currently indexed in the system.
    Call this before query_documents to know what's available.
    """
    client = get_weaviate()
    col    = client.collections.get(COLLECTION)

    # aggregate to get unique doc names
    response = col.aggregate.over_all(total_count=True)
    total    = response.total_count

    # fetch a sample to get doc names
    results = col.query.fetch_objects(
        limit=200,
        return_properties=["doc_name", "page_number"],
    )

    # build doc summary — name + page count
    doc_pages: dict[str, set] = {}
    for obj in results.objects:
        name = obj.properties["doc_name"]
        page = obj.properties["page_number"]
        doc_pages.setdefault(name, set()).add(page)

    docs = [
        {
            "doc_name":   name,
            "page_count": len(pages),
        }
        for name, pages in sorted(doc_pages.items())
    ]

    log.info("list_documents", total_chunks=total, docs=len(docs))

    return {
        "documents":    docs,
        "total_chunks": total,
        "message":      f"{len(docs)} documents indexed with {total} total chunks",
    }


# ── Tool 3: get_document_info ─────────────────────────────────────

@mcp.tool()
def get_document_info(doc_name: str) -> dict:
    """
    Get detailed information about a specific document.
    Returns page count and a content preview from each page.

    Args:
        doc_name: Document name without extension e.g. 'P19-1598'
    """
    import weaviate.classes as wvc

    inp    = DocInfoInput(doc_name=doc_name)
    client = get_weaviate()
    col    = client.collections.get(COLLECTION)

    results = col.query.fetch_objects(
        limit=200,
        filters=wvc.query.Filter.by_property("doc_name").equal(inp.doc_name),
        return_properties=["chunk_text", "page_number", "chunk_index"],
    )

    if not results.objects:
        return {
            "error":    f"Document '{inp.doc_name}' not found in index.",
            "hint":     "Use list_documents to see available documents.",
        }

    # group by page
    pages: dict[int, str] = {}
    for obj in results.objects:
        p    = int(obj.properties["page_number"])
        text = obj.properties["chunk_text"]
        if p not in pages:
            pages[p] = text[:200]  # first 200 chars as preview

    page_previews = [
        {"page": p, "preview": pages[p]}
        for p in sorted(pages.keys())
    ]

    log.info("get_document_info", doc=inp.doc_name, pages=len(pages))

    return {
        "doc_name":      inp.doc_name,
        "page_count":    len(pages),
        "total_chunks":  len(results.objects),
        "page_previews": page_previews,
    }


# ── Startup + shutdown ────────────────────────────────────────────

def on_startup():
    log.info("server_starting", name="nexla-document-qa")

    # run ingestion — skips if already indexed
    summary = run_ingestion_pipeline()
    log.info("ingestion_ready", **summary)

    # warm up Weaviate connection
    get_weaviate()
    log.info("weaviate_connected")
    log.info("server_ready", tools=["query_documents", "list_documents", "get_document_info"])


def on_shutdown():
    global _weaviate_client
    if _weaviate_client:
        _weaviate_client.close()
        log.info("weaviate_closed")

    # print metrics summary
    summary = metrics.summary()
    log.info("metrics_summary", **summary)
    print("\n── Metrics summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("─────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    on_startup()
    try:
        mcp.run(transport="stdio")
    finally:
        on_shutdown()