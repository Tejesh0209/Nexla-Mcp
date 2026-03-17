# Nexla MCP Server — Document Intelligence over PDFs

> **Forward Deployed Engineer – AI · Take-Home Assignment**
> Built by Tejesh Boppana

---

## What this is

An MCP server that lets any AI agent ask natural language questions over 5 academic PDF documents and get back grounded, attributed answers. Every response includes the source document name and page number — no hallucinations, no guessing.

The system uses hybrid retrieval (BM25 + dense vector search) over Weaviate, routes queries to the cheapest model that can handle them via Fireworks AI, and traces every chain through LangSmith. It runs entirely locally with one Docker container.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP CLIENT                               │
│              Claude Desktop / Any MCP-compatible agent          │
└─────────────────────┬───────────────────────────────────────────┘
                      │  stdio  (JSON-RPC)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastMCP Server                               │
│                                                                 │
│   ┌─────────────────┐ ┌──────────────────┐ ┌────────────────┐  │
│   │ query_documents │ │  list_documents  │ │get_document_   │  │
│   │                 │ │                  │ │    info        │  │
│   │ • Pydantic      │ │ • doc names      │ │ • page count   │  │
│   │   validation    │ │ • page counts    │ │ • previews     │  │
│   │ • attribution   │ │ • chunk totals   │ │ • chunk count  │  │
│   └────────┬────────┘ └──────────────────┘ └────────────────┘  │
│            │                                                    │
│   ┌────────▼────────────────────────────────────────────────┐  │
│   │                   Query Router                          │  │
│   │         keyword signals → "rag" or "hybrid"             │  │
│   └────────┬────────────────────┬───────────────────────────┘  │
│            │                    │                               │
│      "rag" │              "hybrid" │                            │
│            ▼                    ▼                               │
│   ┌────────────────┐  ┌──────────────────┐                     │
│   │ Fireworks AI   │  │  Fireworks AI    │                     │
│   │ Mixtral-8x22B  │  │ DeepSeek-v3p2   │                     │
│   │ $0.20/1M tok   │  │  $0.90/1M tok   │                     │
│   └────────┬───────┘  └────────┬─────────┘                     │
│            └────────┬──────────┘                               │
│                     │                                           │
│   ┌─────────────────▼───────────────────────────────────────┐  │
│   │              Weaviate Hybrid Search                     │  │
│   │         BM25 (keyword) + HNSW (semantic)                │  │
│   │                   alpha = 0.75                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                  Observability                          │  │
│   │    LangSmith traces · structlog · metrics.py            │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Weaviate (Docker)                             │
│        DocumentChunk collection · 115 indexed chunks           │
│      BM25 inverted index + HNSW vector index                   │
└─────────────────────────────────────────────────────────────────┘
                      ▲
                      │  Ingestion pipeline (runs once at startup)
                      │
┌─────────────────────────────────────────────────────────────────┐
│  PDFs → PyMuPDF → Chunker (512 tok/64 overlap) →               │
│  OpenAI text-embedding-3-small → Weaviate upsert               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Choice | Why |
|---|---|---|
| MCP Framework | FastMCP | Handles JSON-RPC/stdio boilerplate, auto-generates tool schemas |
| PDF Parsing | PyMuPDF | Fastest, best layout handling on academic papers |
| Chunking | tiktoken cl100k_base | Same tokenizer as embedding model — accurate token counts |
| Embeddings | OpenAI text-embedding-3-small | No real competitor at this price/quality point |
| Vector Store | Weaviate (Docker) | Only local option with native hybrid BM25 + vector search |
| LLM — RAG path | Fireworks Mixtral-8x22B | $0.20/1M tok, excellent at factual Q&A |
| LLM — Hybrid path | Fireworks DeepSeek-v3p2 | $0.90/1M tok, stronger cross-document reasoning |
| Tracing | LangSmith | End-to-end chain visibility with zero extra code |
| Logging | structlog | JSON output, request_id correlation |
| Evaluation | RAGAS | Faithfulness + relevancy + recall + precision |
| Validation | Pydantic v2 | Input guardrails on every tool call |

---

## Why Weaviate over ChromaDB, FAISS, Pinecone

The key insight is that enterprise document Q&A has two types of questions:

- **Semantic** — "what approach did they use for data augmentation?" → needs vector search
- **Keyword** — "what is the BLEU score in Table 2?" → needs exact term matching

ChromaDB gives you dense search only. FAISS is just an index — no metadata storage. Pinecone requires a cloud account. Weaviate runs locally in Docker and fuses BM25 + vector in a single query with a tunable `alpha` parameter. For academic papers with precise terminology, this matters.

---

## Cost Engineering — Fireworks AI vs OpenAI

This was a deliberate architectural decision. The model router sends ~80% of queries to Mixtral and ~20% to DeepSeek based on complexity signals.

**Per query cost (from actual test runs):**

| Path | Model | Tokens | Cost/query |
|---|---|---|---|
| RAG | Fireworks Mixtral-8x22B | 2,898 | $0.00058 |
| Hybrid | Fireworks DeepSeek-v3p2 | 3,068 | $0.00276 |
| **Avg (80/20 split)** | | | **$0.00102** |

**If we had used OpenAI instead:**

| Path | Model | Tokens | Cost/query |
|---|---|---|---|
| RAG | GPT-4o-mini | 2,898 | $0.000495 |
| Hybrid | GPT-4o | 3,068 | $0.00975 |
| **Avg (80/20 split)** | | | **$0.00240** |

**Fireworks is 2.4x cheaper** overall. The saving comes entirely from the hybrid path — GPT-4o charges $10/1M output tokens vs DeepSeek at $0.90/1M. At 10,000 queries/day that is $14/day saved. In an enterprise deployment with 100,000 queries/day it is $140/day — $51,100/year.

The router is the cost architecture. Most questions are factual lookups that do not need a 70B frontier model. Routing them to Mixtral while sending only genuinely complex cross-document questions to DeepSeek is the right engineering call.

---

## Setup

**Prerequisites:** Docker Desktop, Python 3.11+, API keys for OpenAI, Fireworks AI, LangSmith

### 1. Clone and install

```bash
git clone https://github.com/tejeshboppana/nexla-mcp
cd nexla-mcp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys
```

```env
GDRIVE_FOLDER_ID=1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr
OPENAI_API_KEY=your_key
FIREWORKS_API_KEY=your_key
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=nexla-mcp
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
```

### 3. Start Weaviate

```bash
docker compose up -d
curl http://localhost:8080/v1/.well-known/ready   # should return {}
```

### 4. Index the documents

```bash
python3 ingest.py
# Downloads 5 PDFs from Google Drive and indexes them into Weaviate
# Skips automatically if already indexed
# Use --force to re-index from scratch
```

### 5. Run the demo

```bash
python3 test_server.py
```

### 6. Connect to Claude Desktop (optional)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nexla-document-qa": {
      "command": "python3",
      "args": ["/full/path/to/nexla-mcp/src/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. Ask any question — Claude will call `query_documents` automatically.

---

## MCP Tools

### `query_documents`

Ask a natural language question about the indexed documents.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Your question (3–1000 chars) |
| `strategy` | string | `"auto"` | `"auto"` / `"rag"` / `"hybrid"` |
| `top_k` | integer | `5` | Chunks to retrieve (1–20) |

**Returns:**
```json
{
  "answer": "The KGLM achieved perplexity of 44.1...",
  "sources": [
    {"doc_name": "P19-1598", "page_number": 7, "score": 0.9564}
  ],
  "strategy": "rag",
  "model": "mixtral-8x22b-instruct",
  "tokens_used": 2898,
  "latency_ms": 1243.5
}
```

**Example queries:**
- `"What perplexity did KGLM achieve on Linked WikiText-2?"`
- `"Compare the data augmentation strategies used across all papers"`
- `"What is the WinoMT challenge set?"`

---

### `list_documents`

List all documents currently indexed. Call this before querying to know what is available.

No parameters.

**Returns:**
```json
{
  "documents": [
    {"doc_name": "D19-1539", "page_count": 10},
    {"doc_name": "P19-1164", "page_count": 6},
    {"doc_name": "P19-1598", "page_count": 10},
    {"doc_name": "W18-4401", "page_count": 11},
    {"doc_name": "W18-5713", "page_count": 6}
  ],
  "total_chunks": 115,
  "message": "5 documents indexed with 115 total chunks"
}
```

---

### `get_document_info`

Get page count and content preview for a specific document.

| Parameter | Type | Description |
|---|---|---|
| `doc_name` | string | Document name without extension e.g. `"P19-1598"` |

**Returns:**
```json
{
  "doc_name": "P19-1598",
  "page_count": 10,
  "total_chunks": 28,
  "page_previews": [
    {"page": 1, "preview": "Proceedings of the 57th Annual Meeting..."},
    {"page": 2, "preview": "graph, a subset of the knowledge graph..."}
  ]
}
```

---

## Example Interaction Log

### Q1 — Simple factual (RAG path → Mixtral)

**Question:** What perplexity did KGLM achieve on Linked WikiText-2?

**Answer:** KGLM achieved a perplexity of 44.1 on Linked WikiText-2 [P19-1598, page 7].

**Sources:** P19-1598 p7, P19-1598 p2, P19-1598 p6
**Strategy:** rag · Model: mixtral-8x22b-instruct · Tokens: 3,205 · Latency: 2,391ms

---

### Q2 — Factual with attribution (RAG path → Mixtral)

**Question:** What data augmentation strategy did the saroyehun system use?

**Answer:** The saroyehun system used translation as a data augmentation strategy [W18-4401, page 5].

**Sources:** W18-4401 p5, W18-4401 p6
**Strategy:** rag · Model: mixtral-8x22b-instruct · Tokens: 2,679 · Latency: 1,796ms

---

### Q3 — Cross-document (Hybrid path → DeepSeek)

**Question:** Compare the approaches used across all the papers.

**Answer:** Based on the provided context, the papers describe approaches for two distinct tasks. W18-4401 details various system approaches for Aggression Identification including LSTM, CNN, SVM and ensemble methods. The other papers cover knowledge graph language modeling (P19-1598), gender bias in MT (P19-1164), cloze-driven pretraining (D19-1539), and retrieve-and-refine dialogue (W18-5713). A direct cross-task comparison is limited by the retrieved context.

**Sources:** W18-4401 p4, W18-4401 p5, D19-1539 p4
**Strategy:** hybrid · Model: deepseek-v3p2 · Tokens: 3,338 · Latency: 10,289ms

---

### Q4 — BERT training data (RAG path → Mixtral)

**Question:** What training data was used similar to BERT?

**Answer:** The training data used similar to BERT comprises the BooksCorpus of about 800M words plus English Wikipedia data of 2.5B words [D19-1539, page 4].

**Sources:** D19-1539 p4, D19-1539 p2
**Strategy:** rag · Model: mixtral-8x22b-instruct · Tokens: 2,875 · Latency: 2,252ms

---

### Q5 — Exact number lookup (RAG path → Mixtral)

**Question:** How many instances does the WinoMT challenge set contain?

**Answer:** The WinoMT challenge set contains 3,888 instances [P19-1164, page 2].

**Sources:** P19-1164 p2, P19-1164 p5
**Strategy:** rag · Model: mixtral-8x22b-instruct · Tokens: 2,827 · Latency: 1,740ms

---

## RAGAS Evaluation

Offline evaluation on 15 hand-crafted questions grounded in the actual PDFs.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.9571** | Answer grounded in retrieved context |
| Answer Relevancy | **0.8533** | Answer addresses the question |
| Context Recall | **0.9394** | Retrieval found the right chunks |
| Context Precision | **0.7823** | Low noise in retrieved set |

Faithfulness (0.96) and context recall (0.94) are the strongest scores — the grounding constraint in the prompt is working and Weaviate hybrid search is finding the right chunks. Context precision (0.78) is the lowest score and the expected one — at top_k=5, ~22% of retrieved chunks are topically related but not directly useful for the specific question. Reducing to top_k=3 would raise precision but lower recall. 0.78 is acceptable for production document Q&A.

*Evaluated offline on 15 hand-crafted questions grounded in the actual PDFs. Run `python3 eval/ragas_eval.py` to reproduce. Takes ~3 minutes, costs ~$0.10.*

---

## Observability

| Signal | Tool | What is captured |
|---|---|---|
| Traces | LangSmith | Full chain: router → retrieval → synthesis, per-step latency, token cost |
| Metrics | metrics.py | Latency p50/p99, avg tokens, route distribution |
| Logs | structlog | JSON events with request_id correlation |

**Metrics from test run (5 queries):**

```
Queries run        : 5
Avg tokens/query   : 2,984
Route distribution : {rag: 4, hybrid: 1}  — 80/20 exactly as designed
p50 latency        : 2,252ms
p99 latency        : 10,289ms  (DeepSeek hybrid queries)
```

RAG queries (Mixtral) average ~2.2s. Hybrid queries (DeepSeek) average ~10s due to model size. The router sends 80% of queries to the faster, cheaper path.

---

## Architectural Decisions and Trade-offs

### Why no CAG (Cache-Augmented Generation)

During vibe coding, Claude Code suggested implementing CAG alongside RAG — loading entire documents into the LLM context window for broad summarization questions. I rejected this for three concrete reasons.

**First, cost.** The 5 PDFs total roughly 165k tokens. A single CAG query would cost approximately $0.41 at Claude Sonnet pricing. The reviewer runs 10 test queries — that is $4 just on CAG calls. The assignment asks for a system that "scales." A query that costs 400x more than a RAG query does not scale.

**Second, redundancy.** Weaviate hybrid search with alpha=0.75 already handles the broad question case well. When a question like "compare approaches across all papers" fires the hybrid route, DeepSeek receives 5 well-chosen chunks spanning multiple documents. The answer quality is equivalent to CAG at 1/400th the cost. CAG only wins when retrieval fails — and with hybrid search, retrieval rarely fails on these documents.

**Third, complexity without payoff.** CAG would have added a third model path, a third branch in the router, and a Claude API dependency on top of Fireworks. For a local take-home demo, that is three more things that can break on the reviewer's machine. I chose to make the system more reliable rather than more impressive on paper.

The right call for this corpus: hybrid RAG with a strong retrieval layer beats CAG at a fraction of the cost.

### Why Fireworks over OpenAI for inference

Documented in the cost section above. The short version: 2.4x cheaper, no meaningful quality difference for factual document Q&A, and using a non-OpenAI provider demonstrates stack diversity — which is relevant for a Forward Deployed Engineer who needs to advise customers on vendor selection.

### Why the query router uses keyword heuristics not an LLM classifier

An LLM-based router would add one round-trip API call before every query — 400-600ms of extra latency and $0.0003 per call. At 10,000 queries/day that is $3/day just for routing. The keyword heuristic (check for "compare", "across all", "difference" etc.) achieves equivalent routing quality for these document types with zero cost and zero latency. The right tool for the job is not always the most sophisticated one.

---

## Vibe Coding Section

**Tools used:** Claude Code (primary), Claude claude.ai for architecture discussions

### How I used Claude Code

I used Claude Code throughout the build but in a specific way — I treated it as a senior pair programmer who writes fast but needs architectural direction from me.

The workflow was: I designed the architecture first (data flow, module boundaries, which tools at each layer), then used Claude Code to scaffold the implementation of each module. I gave it detailed context on what each file needed to do, what data structures to pass between layers, and what the constraints were.

### What Claude Code got right

Boilerplate was near-perfect. The Weaviate schema definition, the tiktoken chunking loop, the OpenAI embeddings batching with retry logic, the structlog configuration — Claude Code wrote all of these faster and more correctly than I would have written them manually. It also caught edge cases I would have missed, like the `dominant_baseline="central"` SVG attribute and the `remaining_ok=True` flag in gdown.

### What I overrode

**The entire architecture.** Claude Code's initial suggestion was: single LLM (Claude), ChromaDB for vector storage, one `query_documents` tool, basic dense retrieval. That is the default RAG pattern and it would have been a forgettable submission.

I replaced every layer: Weaviate for hybrid search instead of ChromaDB, Fireworks AI with a model router instead of a single Claude call, three specialized tools instead of one generic tool, and a query router that makes cost-aware decisions. Claude Code helped me implement the architecture I designed — it did not design the architecture.

**CAG.** Claude Code repeatedly suggested adding CAG as a third retrieval path. I rejected it for the reasons documented above. This is a case where the AI saw an interesting technical pattern and wanted to implement it. I evaluated the trade-offs and decided the cost/complexity did not justify the marginal quality improvement.

**The prompt template.** Claude Code's first synthesizer prompt was verbose and included few-shot examples. I stripped it to zero-shot with explicit grounding constraints. For factual document Q&A, zero-shot with a strong grounding instruction outperforms few-shot in both quality and token efficiency.

### What I learned

Claude Code is genuinely useful for implementation velocity — modules that would take me 45 minutes to write carefully took 5 minutes with Claude Code doing the first draft. But the value of AI tooling in forward-deployed work is not replacing engineering judgment. It is compressing the time between architectural decisions and working code.

The engineers who get the most from these tools are the ones who have strong enough architectural judgment to know when the AI's suggestion is good and when it is optimizing for the wrong thing. Without that judgment, you end up with a technically impressive system that does not actually solve the problem well.

### Forward-deployed engineering mindset

In a customer engagement, I would use Claude Code the same way — to move fast on implementation while keeping architectural decisions in my hands. The customer does not care how fast I typed the code. They care whether the system solves their problem reliably at a cost that makes sense. AI tooling helps with speed. Engineering judgment determines whether you are building the right thing.

---

## Project Structure

```
nexla-mcp/
├── src/
│   ├── ingestion/
│   │   ├── downloader.py       # gdown from Google Drive
│   │   ├── parser.py           # PyMuPDF page extraction
│   │   ├── chunker.py          # tiktoken 512/64 overlap
│   │   ├── embedder.py         # OpenAI text-embedding-3-small
│   │   ├── weaviate_store.py   # schema + batch upsert
│   │   └── pipeline.py         # orchestrates all 5 steps
│   ├── retrieval/
│   │   ├── retriever.py        # Weaviate hybrid search
│   │   └── router.py           # query complexity classifier
│   ├── synthesis/
│   │   └── synthesizer.py      # Fireworks AI + prompt template
│   ├── observability/
│   │   ├── logging_config.py   # structlog setup
│   │   └── metrics.py          # in-memory metrics collector
│   └── mcp_server.py           # FastMCP + 3 tools
├── eval/
│   └── ragas_eval.py           # offline evaluation script
├── data/raw/                   # PDFs (gitignored)
├── ingest.py                   # CLI ingestion entry point
├── test_server.py              # demo script (no Claude Desktop needed)
├── docker-compose.yml          # Weaviate
├── requirements.txt
├── .env.example
└── README.md
```

---

## What I would add with more time

- **LangGraph orchestration** — multi-step agent that calls `list_documents` first, then routes to `query_documents` or `get_document_info` based on the question type
- **CAG for very broad summaries** — once the corpus grows beyond 20 documents where retrieval becomes less reliable
- **Prometheus + Grafana** — replace metrics.py with proper time-series metrics
- **Re-ranking** — add a cross-encoder reranker between retrieval and synthesis to improve context precision
- **Streaming responses** — FastMCP supports streaming; adding it would improve perceived latency on long answers

---

## Notes on the assignment scope

The assignment estimated 3-4 hours. I went over that deliberately. The core requirement (one working `query_documents` tool) could be built in 2 hours. The additional decisions — hybrid search, model router, cost comparison, RAGAS evaluation, three specialized tools — each added time but each also demonstrated a specific engineering judgment call. I documented every trade-off because the assignment says "we are looking for how you think." The README is part of the submission, not an afterthought.

---

*Built with Python 3.11 · FastMCP · Weaviate · Fireworks AI · LangSmith · RAGAS*