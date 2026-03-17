"""
test_server.py
Standalone demo of all 3 MCP tools — no Claude Desktop needed.
Uses real questions grounded in the 5 indexed PDFs.

Usage:
    python3 test_server.py
"""

import os
import sys
import time

import structlog
import logging
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pipeline import run_ingestion_pipeline
from src.ingestion.weaviate_store import get_client, COLLECTION
from src.retrieval.router import classify
from src.retrieval.retriever import hybrid_search
from src.synthesis.synthesizer import synthesize
from src.observability.metrics import metrics


def sep(title=""):
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    else:
        print(f"\n{'─' * 60}")


def run_query(client, question, strategy="auto", top_k=5):
    start    = time.time()
    route    = classify(question) if strategy == "auto" else strategy
    chunks   = hybrid_search(client, question, top_k=top_k)
    answer   = synthesize(question, chunks, route)
    elapsed  = round((time.time() - start) * 1000, 1)
    metrics.record(route, answer.tokens_used, elapsed)
    return answer, route, elapsed, chunks


def tool_list_documents(client):
    col     = client.collections.get(COLLECTION)
    count   = col.aggregate.over_all(total_count=True).total_count
    results = col.query.fetch_objects(
        limit=200,
        return_properties=["doc_name", "page_number"],
    )
    doc_pages: dict = {}
    for obj in results.objects:
        name = obj.properties["doc_name"]
        page = obj.properties["page_number"]
        doc_pages.setdefault(name, set()).add(page)

    docs = [
        {"doc_name": name, "page_count": len(pages)}
        for name, pages in sorted(doc_pages.items())
    ]
    return docs, count


def tool_get_document_info(client, doc_name):
    import weaviate.classes as wvc
    col     = client.collections.get(COLLECTION)
    results = col.query.fetch_objects(
        limit=200,
        filters=wvc.query.Filter.by_property("doc_name").equal(doc_name),
        return_properties=["chunk_text", "page_number"],
    )
    pages = {}
    for obj in results.objects:
        p    = int(obj.properties["page_number"])
        text = obj.properties["chunk_text"]
        if p not in pages:
            pages[p] = text[:200]
    return pages, len(results.objects)


def main():
    print("\n" + "=" * 60)
    print("  Nexla MCP Server — Demo")
    print("  5 PDFs · Hybrid RAG · Fireworks AI")
    print("=" * 60)

    # ── STEP 1: Ingestion ─────────────────────────────────────────
    sep("Step 1: Ingestion pipeline")
    summary = run_ingestion_pipeline()
    print(f"  Status  : {summary['status']}")
    print(f"  PDFs    : {summary['pdfs']}")
    if summary.get("chunks"):
        print(f"  Chunks  : {summary['chunks']}")

    client = get_client()

    # ── TOOL 1: list_documents ────────────────────────────────────
    sep("Tool: list_documents")
    docs, total_chunks = tool_list_documents(client)
    print(f"  {len(docs)} documents indexed · {total_chunks} total chunks\n")
    for d in docs:
        print(f"    • {d['doc_name']:<20} ({d['page_count']} pages)")

    # ── TOOL 2: get_document_info ─────────────────────────────────
    sep("Tool: get_document_info  →  P19-1598")
    pages, chunk_count = tool_get_document_info(client, "P19-1598")
    print(f"  Document    : P19-1598")
    print(f"  Pages       : {len(pages)}")
    print(f"  Chunks      : {chunk_count}")
    print(f"  Page 1 preview:")
    print(f"    {list(pages.values())[0][:150]}...")

    # ── TOOL 3: query_documents — real questions from the PDFs ────
    sep("Tool: query_documents — 5 sample questions")

    questions = [
        # RAG — simple factual from P19-1598
        "What perplexity did KGLM achieve on Linked WikiText-2?",
        # RAG — factual from W18-4401
        "What data augmentation strategy did the saroyehun system use?",
        # HYBRID — cross-document comparison
        "Compare the approaches used across all the papers.",
        # RAG — factual from D19-1539
        "What training data was used similar to BERT?",
        # RAG — factual from P19-1164
        "How many instances does the WinoMT challenge set contain?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n  [{i}] {q}")
        answer, route, elapsed, chunks = run_query(client, q)
        top_src = f"{chunks[0].doc_name} p{chunks[0].page_number}" if chunks else "n/a"
        print(f"       Route   : {route:<8} Model : {answer.model}")
        print(f"       Tokens  : {answer.tokens_used:<6}  Latency : {elapsed}ms")
        print(f"       Top src : {top_src}")
        print(f"       Answer  : {answer.text[:220]}{'...' if len(answer.text) > 220 else ''}")
        print(f"       Sources : {', '.join(s['doc_name'] + ' p' + str(s['page_number']) for s in answer.sources[:3])}")

    # ── Metrics Summary ───────────────────────────────────────────
    sep("Metrics summary")
    s = metrics.summary()
    print(f"  Queries run        : {s['queries']}")
    print(f"  Avg tokens/query   : {s['avg_tokens']}")
    print(f"  Route distribution : {s['route_distribution']}")
    print(f"  p50 latency        : {s['p50_latency_ms']} ms")
    print(f"  p99 latency        : {s['p99_latency_ms']} ms")

    client.close()

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()