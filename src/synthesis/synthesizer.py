import os
from dataclasses import dataclass

import fireworks.client
import structlog

from src.retrieval.retriever import RetrievedChunk

log = structlog.get_logger(__name__)

# Mixtral for simple RAG — fast, cheap, great at factual lookups
RAG_MODEL    = "accounts/fireworks/models/mixtral-8x22b-instruct"

# DeepSeek for hybrid — stronger cross-document reasoning
HYBRID_MODEL = "accounts/fireworks/models/deepseek-v3p2"

MAX_TOKENS   = 1024
TEMPERATURE  = 0.1  # keep it factual, not creative


@dataclass
class Answer:
    text:        str
    sources:     list[dict]  # [{doc_name, page_number, score}]
    strategy:    str         # "rag" or "hybrid"
    model:       str         # short model name for logging
    tokens_used: int


def _build_context(chunks: list[RetrievedChunk]) -> str:
    # number each chunk and label it with source so the model can cite it
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c.doc_name}, Page {c.page_number}\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return f"""You are a precise research assistant answering questions about academic papers.
Answer using ONLY the provided context. Always cite sources using [doc_name, page X].
If the context does not contain enough information to answer, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def synthesize(
    question: str,
    chunks:   list[RetrievedChunk],
    strategy: str,
) -> Answer:
    """
    Generate a grounded answer from retrieved chunks.
    Picks Mixtral for simple RAG queries, DeepSeek for complex hybrid queries.
    Returns Answer with full source attribution.
    """
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("FIREWORKS_API_KEY is not set")

    fireworks.client.api_key = api_key

    model   = HYBRID_MODEL if strategy == "hybrid" else RAG_MODEL
    context = _build_context(chunks)
    prompt  = _build_prompt(question, context)

    log.info(
        "synthesizing",
        strategy = strategy,
        model    = model.split("/")[-1],
        chunks   = len(chunks),
    )

    response = fireworks.client.ChatCompletion.create(
        model      = model,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = MAX_TOKENS,
        temperature = TEMPERATURE,
    )

    answer_text = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    sources = [
        {
            "doc_name":    c.doc_name,
            "page_number": c.page_number,
            "score":       round(c.score, 4),
        }
        for c in chunks
    ]

    log.info(
        "synthesis_done",
        tokens = tokens_used,
        model  = model.split("/")[-1],
    )

    return Answer(
        text        = answer_text,
        sources     = sources,
        strategy    = strategy,
        model       = model.split("/")[-1],
        tokens_used = tokens_used,
    )