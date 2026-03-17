"""
ragas_eval.py
Offline evaluation of the retrieval + synthesis pipeline using RAGAS.
Measures 4 metrics: faithfulness, answer relevancy, context recall, context precision.
Questions are grounded in the actual 5 PDFs — no fabricated ground truths.
Run once after indexing. Paste scores into README observability table.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import structlog
import logging

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    processors=[structlog.dev.ConsoleRenderer(colors=True)],
)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.ingestion.weaviate_store import get_client
from src.retrieval.retriever import hybrid_search
from src.retrieval.router import classify
from src.synthesis.synthesizer import synthesize


# 15 questions grounded directly in the 5 indexed PDFs
EVAL_QUESTIONS = [

    # ── P19-1598: Knowledge Graph Language Model ──────────────────
    {
        "question": "What is the KGLM and what problem does it solve?",
        "ground_truth": (
            "KGLM stands for Knowledge Graph Language Model. It is a neural "
            "language model with mechanisms for selecting and copying facts from "
            "a knowledge graph relevant to the context. It solves the problem that "
            "traditional language models can only memorize facts seen at training "
            "time and struggle to generate factually correct text about rare entities."
        ),
    },
    {
        "question": "What is the Linked WikiText-2 dataset?",
        "ground_truth": (
            "Linked WikiText-2 is a corpus of annotated text aligned to the Wikidata "
            "knowledge graph whose contents roughly match the popular WikiText-2 "
            "benchmark. Tokens in the text are linked to entities in Wikidata using "
            "human-provided links and off-the-shelf linking and coreference models."
        ),
    },
    {
        "question": "What perplexity did KGLM achieve compared to AWD-LSTM?",
        "ground_truth": (
            "KGLM achieved a perplexity of 44.1 and unknown-penalized perplexity "
            "of 88.5, compared to AWD-LSTM which achieved perplexity of 74.8 and "
            "unknown-penalized perplexity of 165.8 on Linked WikiText-2."
        ),
    },
    {
        "question": "What embedding method does KGLM use for entities and relations?",
        "ground_truth": (
            "KGLM uses fixed entity and relation embeddings pre-trained using TransE "
            "on Wikidata. Given a triple (p, r, e), it learns embeddings vp, vr and ve "
            "to minimize the TransE distance using a max-margin loss."
        ),
    },

    # ── W18-4401: TRAC Aggression Identification Shared Task ──────
    {
        "question": "What were the three categories of aggression in the TRAC shared task?",
        "ground_truth": (
            "The three categories were Overtly Aggressive (OAG), "
            "Covertly Aggressive (CAG), and Non-Aggressive (NAG)."
        ),
    },
    {
        "question": "What was the best weighted F-score achieved in the TRAC shared task?",
        "ground_truth": (
            "The best system obtained a weighted F-score of 0.64 for both Hindi and "
            "English on the Facebook test sets. The best scores on the surprise set "
            "were 0.60 and 0.50 for English and Hindi respectively."
        ),
    },
    {
        "question": "What data augmentation strategy did the saroyehun system use?",
        "ground_truth": (
            "The saroyehun system used translation as a data augmentation strategy "
            "and pseudolabelled the dataset using an external dataset on hate speech. "
            "It was the only system that performed better on the Twitter dataset "
            "despite being trained on the Facebook dataset."
        ),
    },
    {
        "question": "How many teams registered and submitted systems in the TRAC shared task?",
        "ground_truth": (
            "A total of 130 teams registered to participate, 30 teams submitted their "
            "test runs, and 20 teams also sent their system description papers."
        ),
    },

    # ── P19-1164: Gender Bias in Machine Translation ──────────────
    {
        "question": "What is the WinoMT challenge set?",
        "ground_truth": (
            "WinoMT is a challenge set for gender bias evaluation in machine translation "
            "composed by concatenating the Winogender and WinoBias coreference test sets. "
            "It contains 3,888 instances equally balanced between male and female genders "
            "and between stereotypical and non-stereotypical gender-role assignments."
        ),
    },
    {
        "question": "What did the gender bias study find about commercial MT systems?",
        "ground_truth": (
            "All four tested commercial MT systems (Google Translate, Microsoft Translator, "
            "Amazon Translate, SYSTRAN) and two academic models were significantly prone to "
            "gender-biased translation errors for all tested target languages. All systems "
            "performed better on stereotypical gender role assignments than non-stereotypical ones."
        ),
    },
    {
        "question": "What eight target languages were tested for gender bias in machine translation?",
        "ground_truth": (
            "The eight target languages tested were Spanish, French, Italian, Russian, "
            "Ukrainian, Hebrew, Arabic, and German."
        ),
    },

    # ── D19-1539: Cloze-driven Pretraining ───────────────────────
    {
        "question": "What is the cloze-driven pretraining objective in the two tower model?",
        "ground_truth": (
            "The model solves a cloze-style word reconstruction task where each word is "
            "ablated and must be predicted given the rest of the text. The model separately "
            "computes forward and backward states using masked self-attention then combines "
            "them to jointly predict the center word."
        ),
    },
    {
        "question": "What training data similar to BERT was used in the cloze pretraining experiments?",
        "ground_truth": (
            "BooksCorpus plus English Wikipedia was used, similar to BERT. The BooksCorpus "
            "contains about 800M words and English Wikipedia data contains 2.5B words."
        ),
    },
    {
        "question": "How did the cloze objective compare to the bilm objective on GLUE?",
        "ground_truth": (
            "The cloze loss performed significantly better than the bilm loss on GLUE "
            "development sets, achieving an average score of 80.9 versus 79.3 for bilm. "
            "Combining the two loss types did not improve over the cloze loss alone."
        ),
    },

    # ── W18-5713: Retrieve and Refine ─────────────────────────────
    {
        "question": "How does the Retrieve and Refine model work?",
        "ground_truth": (
            "The Retrieve and Refine model first retrieves a response using a Key-Value "
            "Memory Network retrieval model, then feeds the retrieved utterance concatenated "
            "with the dialogue history to a standard Seq2Seq generator which refines it. "
            "The retrieved utterance is prepended with a special separator token."
        ),
    },
]


def run_eval():
    print("\n" + "=" * 60)
    print("  RAGAS Offline Evaluation")
    print("=" * 60)
    print(f"  Questions : {len(EVAL_QUESTIONS)}")
    print(f"  Metrics   : faithfulness, answer_relevancy,")
    print(f"              context_recall, context_precision")
    print("=" * 60 + "\n")

    client = get_client()

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i:2}/{len(EVAL_QUESTIONS)}] {q[:65]}...")

        try:
            strategy = classify(q)
            chunks   = hybrid_search(client, q, top_k=5)
            answer   = synthesize(q, chunks, strategy)

            questions.append(q)
            answers.append(answer.text)
            contexts.append([c.text for c in chunks])
            ground_truths.append(gt)

        except Exception as e:
            print(f"           ERROR: {e}")
            continue

    client.close()

    total = len(questions)
    print(f"\n  Completed {total}/{len(EVAL_QUESTIONS)} questions")
    print("  Building RAGAS dataset...")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    print("  Running RAGAS scoring (2-3 minutes)...\n")

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ChatOpenAI(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    print("\n" + "=" * 60)
    print("  RAGAS RESULTS")
    print("=" * 60)
    def _mean(val):
        if isinstance(val, list):
            return sum(v for v in val if v is not None) / max(len([v for v in val if v is not None]), 1)
        return val

    print(f"  Faithfulness      : {_mean(results['faithfulness']):.4f}")
    print(f"  Answer Relevancy  : {_mean(results['answer_relevancy']):.4f}")
    print(f"  Context Recall    : {_mean(results['context_recall']):.4f}")
    print(f"  Context Precision : {_mean(results['context_precision']):.4f}")
    print("=" * 60)
    print("\n  Paste these scores into your README observability table.")
    print("  Note: eval run on offline sample of 15 questions.\n")

    return results


if __name__ == "__main__":
    run_eval()