import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import pymupdf as fitz

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.hybrid_engine import hybrid_engine, tokenize
from app.schemas.rag import ChunkResult


SAMPLE_DOCS_DIR = "backend/data/sample_docs"
SAMPLE_PDF_PATH = os.path.join(SAMPLE_DOCS_DIR, "datascout_system_architecture.pdf")
BENCHMARK_RESULTS_PATH = "backend/data/benchmark_results.json"


# 12 Realistic technical questions with target matching keywords or exact phrases in ground-truth chunks
BENCHMARK_DATASET = [
    {
        "id": "q1",
        "question": "What is the exact mathematical formula used by Reciprocal Rank Fusion?",
        "ground_truth_keywords": ["rrf(d)", "1 / (60 + rank)", "reciprocal rank fusion formula"]
    },
    {
        "id": "q2",
        "question": "What does error code ERR-701 indicate in DataScout?",
        "ground_truth_keywords": ["err-701", "index synchronization fault"]
    },
    {
        "id": "q3",
        "question": "How are PDF documents segmented during text ingestion?",
        "ground_truth_keywords": ["500-character", "50-character overlap", "sliding-window"]
    },
    {
        "id": "q4",
        "question": "Which specific vector embedding model is utilized for semantic search?",
        "ground_truth_keywords": ["gemini-embedding-001", "embedding_model", "dense vector"]
    },
    {
        "id": "q5",
        "question": "What keyword matching algorithm powers the lexical search component?",
        "ground_truth_keywords": ["bm25okapi", "rank_bm25", "lexical keyword"]
    },
    {
        "id": "q6",
        "question": "What causes error code ERR-802 and how is it resolved?",
        "ground_truth_keywords": ["err-802", "chromadb connection timeout", "heartbeat retry"]
    },
    {
        "id": "q7",
        "question": "What happens when retrieved chunks fail the relevance grading threshold?",
        "ground_truth_keywords": ["rewrite_query", "retry_count", "self-correction", "technical recall"]
    },
    {
        "id": "q8",
        "question": "When does DataScout trigger live web search fallback?",
        "ground_truth_keywords": ["web_fallback", "tavily", "retry_count >= 1"]
    },
    {
        "id": "q9",
        "question": "What citation format is strictly enforced for knowledge base answers?",
        "ground_truth_keywords": ["[filename · p.x]", "inline citations", "grounded citation"]
    },
    {
        "id": "q10",
        "question": "What is the function of error code ERR-905 in the telemetry pipeline?",
        "ground_truth_keywords": ["err-905", "gemini quota exhausted", "rate limit backoff"]
    },
    {
        "id": "q11",
        "question": "Where is the persistent vector database stored locally on disk?",
        "ground_truth_keywords": ["backend/data/chroma_db", "chroma_persist_dir", "persistentclient"]
    },
    {
        "id": "q12",
        "question": "How does the lightweight query cache improve retrieval latency?",
        "ground_truth_keywords": ["query_cache", "0ms latency", "cache_hit_retrieval"]
    }
]


def create_sample_pdf_if_needed():
    """Generates a rich, multi-page technical manual if no documents exist."""
    if os.path.exists(SAMPLE_PDF_PATH) and hybrid_engine.chunks_store:
        return

    os.makedirs(SAMPLE_DOCS_DIR, exist_ok=True)
    doc = fitz.open()

    pages_content = [
        # Page 1
        "DataScout Engineering Manual - Section 1: Ingestion & Sliding Window Chunking\n\n"
        "PDF documents uploaded to DataScout are processed via PyMuPDF. Text is extracted page-by-page "
        "and partitioned into 500-character chunks with a 50-character overlap using a sliding-window technique. "
        "Each chunk maintains strict metadata tracking its source filename, page number, and unique chunk_id. "
        "This ensures granular traceability and auditability for every retrieved passage.",

        # Page 2
        "DataScout Engineering Manual - Section 2: Hybrid Retrieval & RRF Math\n\n"
        "DataScout combines dense semantic search and sparse lexical search into a single unified rank list. "
        "The keyword matching algorithm is BM25Okapi from rank_bm25, ideal for identifiers and exact terms. "
        "Semantic search uses ChromaDB with gemini-embedding-001 stored at backend/data/chroma_db. "
        "The Reciprocal Rank Fusion formula is defined as: RRF(d) = sum(1 / (60 + rank_r(d))), where k=60. "
        "This prevents any single retriever from dominating top-ranked results.",

        # Page 3
        "DataScout Engineering Manual - Section 3: System Error Codes & Diagnostics\n\n"
        "DataScout monitors system telemetry with standard diagnostic codes:\n"
        "- ERR-701: Index synchronization fault between BM25 memory corpus and ChromaDB collection.\n"
        "- ERR-802: ChromaDB connection timeout; triggered when heartbeat retry fails after 3000ms.\n"
        "- ERR-905: Gemini quota exhausted; triggers exponential rate limit backoff and status warning.\n"
        "Engineers must review the trace_steps log when diagnosing ERR-701 and ERR-802 issues.",

        # Page 4
        "DataScout Engineering Manual - Section 4: Self-Correcting Agent & Tavily Web Fallback\n\n"
        "When retrieved chunks fail the relevance grading threshold, the agent transitions to rewrite_query, "
        "rewriting the query for improved technical recall and incrementing retry_count. "
        "When retry_count >= 1 and chunks still fail relevance grading, DataScout triggers web_fallback "
        "using Tavily live search. Grounded answers cite internal docs as [filename · p.X] or web as [Title](URL).",

        # Page 5
        "DataScout Engineering Manual - Section 5: Telemetry, Cache & Latency Profiling\n\n"
        "An in-memory query_cache stores verified query answers. Repeated queries return instantly with "
        "0ms latency and append cache_hit_retrieval to the trace steps. Full telemetry records rrf_score, "
        "bm25_rank, vector_rank, is_relevant, latency_ms, and cache_hit flags for transparent auditing."
    ]

    for content in pages_content:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, content)

    doc.save(SAMPLE_PDF_PATH)
    doc.close()
    print(f"Created sample technical manual: {SAMPLE_PDF_PATH}")

    # Index into hybrid engine
    with open(SAMPLE_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    chunks = hybrid_engine.extract_and_chunk_pdf(pdf_bytes, os.path.basename(SAMPLE_PDF_PATH))
    indexed = hybrid_engine.index_chunks(chunks)
    print(f"Indexed {indexed} chunks from sample manual into ChromaDB and BM25Okapi.")


def evaluate_retriever_mode(mode: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluates vector_only, bm25_only, or hybrid_rrf across the 12 benchmark questions.
    Computes Recall@5, Precision@5, MRR, and P95 Latency.
    """
    recalls = []
    precisions = []
    reciprocal_ranks = []
    latencies = []

    for item in BENCHMARK_DATASET:
        q = item["question"]
        keywords = [kw.lower() for kw in item["ground_truth_keywords"]]

        start_time = time.perf_counter()

        if mode == "vector_only":
            # Vector only retrieval
            results_raw = hybrid_engine.search_vector(q, top_k=top_k)
            retrieved_chunk_ids = [cid for cid, _, _ in results_raw]
        elif mode == "bm25_only":
            # BM25 only retrieval
            results_raw = hybrid_engine.search_bm25(q, top_k=top_k)
            retrieved_chunk_ids = [cid for cid, _, _ in results_raw]
        elif mode == "hybrid_rrf":
            # Hybrid RRF
            results_chunks = hybrid_engine.hybrid_search(q, top_k=top_k)
            retrieved_chunk_ids = [c.chunk_id for c in results_chunks]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        # Retrieve text of returned chunks
        chunk_lookup = {c["chunk_id"]: c["text"].lower() for c in hybrid_engine.chunks_store}
        retrieved_texts = [chunk_lookup.get(cid, "") for cid in retrieved_chunk_ids]

        # Calculate hits: chunk contains any of the target ground_truth_keywords
        hits = [
            any(kw in text for kw in keywords)
            for text in retrieved_texts
        ]

        relevant_retrieved = sum(hits)
        precision = relevant_retrieved / top_k if top_k > 0 else 0.0
        # For our 1-passage ground truth per specific topic, target_relevant = 1
        recall = 1.0 if relevant_retrieved > 0 else 0.0

        # Reciprocal Rank: 1 / rank of first relevant result
        rr = 0.0
        for rank_idx, is_hit in enumerate(hits):
            if is_hit:
                rr = 1.0 / (rank_idx + 1)
                break

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    return {
        "recall_at_5": round(float(np.mean(recalls)), 4),
        "precision_at_5": round(float(np.mean(precisions)), 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2)
    }


def run_benchmark():
    create_sample_pdf_if_needed()

    modes = ["vector_only", "bm25_only", "hybrid_rrf"]
    results = {
        "benchmark_dataset_size": len(BENCHMARK_DATASET),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {}
    }

    print("\n" + "=" * 78)
    print("  DataScout Retrieval Benchmark — 12 Technical Q&A Test Suite")
    print("=" * 78)
    header = f"{'Retrieval Mode':<16} | {'Recall@5':<10} | {'Precision@5':<12} | {'MRR':<8} | {'P95 Latency':<12}"
    print(header)
    print("-" * 78)

    for mode in modes:
        res = evaluate_retriever_mode(mode, top_k=5)
        results["metrics"][mode] = res
        print(f"{mode:<16} | {res['recall_at_5']:<10.4f} | {res['precision_at_5']:<12.4f} | {res['mrr']:<8.4f} | {res['p95_latency_ms']} ms")

    print("=" * 78)

    # Save to JSON
    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_PATH), exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {BENCHMARK_RESULTS_PATH}\n")


if __name__ == "__main__":
    run_benchmark()
