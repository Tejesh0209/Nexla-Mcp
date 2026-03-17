import re
import structlog

log = structlog.get_logger(__name__)

# questions that need cross-document reasoning → Qwen2.5-72B
HYBRID_SIGNALS = [
    "compare", "difference", "across", "all documents", "both",
    "which paper", "multiple", "each", "between", "contrast",
    "all papers", "summarize all", "overall", "in general",
]

# questions that are simple factual lookups → Llama-3.1-8B
RAG_SIGNALS = [
    "what is", "what are", "who", "when", "how many", "define",
    "list", "name", "which", "where", "what does",
]


def classify(question: str) -> str:
    """
    Classify a question into 'rag' or 'hybrid'.

    rag    → simple factual lookup, single document
             → Fireworks Llama-3.1-8B  ($0.20/1M tokens)

    hybrid → cross-document, comparative, analytical
             → Fireworks Qwen2.5-72B   ($0.90/1M tokens)

    Default is 'rag' — we only escalate when there's a clear signal.
    This keeps the average cost low since most questions are factual.
    """
    q = question.lower().strip()

    # explicit cross-document signals escalate to hybrid
    for signal in HYBRID_SIGNALS:
        if signal in q:
            log.info("route_decision", strategy="hybrid", signal=signal)
            return "hybrid"

    # everything else is rag — cheap and fast
    log.info("route_decision", strategy="rag", question=question[:60])
    return "rag"