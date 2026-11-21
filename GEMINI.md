# VeriRAG — Complete Project Blueprint & Engineering Rules

## 1. Project Overview
VeriRAG is a self-correcting, auditable Hybrid RAG engineering workbench built for an AI Engineering hackathon and portfolio. Unlike basic chatbots that blindly trust vector similarity, VeriRAG audits and proves its retrieval quality using Hybrid Search (BM25 + Vector + Reciprocal Rank Fusion), LangGraph self-correction, a live web search fallback when internal knowledge base (KB) results are weak, and quantitative retrieval benchmarks (Recall@5, Precision@5, MRR, P95 Latency).

## 2. Hackathon Architecture & Stack
- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **PDF Ingestion:** `PyMuPDF` (`fitz`), 500-character sliding-window chunks with 50-character overlap, preserving `source` filename, `page` number, and `chunk_id` metadata
- **Hybrid Retrieval Engine:**
  - **Keyword Search:** `rank_bm25` (`BM25Okapi`) for exact technical terms, error codes, and identifiers[cite: 1]
  - **Semantic Search:** `ChromaDB` (persistent local folder `backend/data/chroma_db`) using Google Gemini `text-embedding-004`[cite: 1]
  - **Rank Fusion:** Reciprocal Rank Fusion (`RRF(d) = sum(1 / (60 + rank_r(d)))`), explicitly returning `rrf_score`, `bm25_rank`, and `vector_rank` per chunk[cite: 1]
- **Corrective Agent Pipeline (LangGraph):**
  1. `retrieve_kb`: Run Hybrid RRF search on uploaded PDFs[cite: 1]
  2. `grade_relevance`: Use `gemini-2.5-flash` to judge if retrieved chunks actually answer the query[cite: 1]
  3. `rewrite_query`: If KB relevance is weak (first attempt), rewrite the query for better technical recall and retry `retrieve_kb` once[cite: 1]
  4. `web_fallback`: If KB results remain weak after retry, search the live web via `duckduckgo-search` (or Tavily) and synthesize a web-grounded answer clearly tagged as `source_mode="web_fallback"` with source URLs
  5. `generate_answer`: Synthesize a grounded answer with inline citations `[filename · p.X]`[cite: 1]
- **Lightweight Hackathon Storage & Cache:** In-memory Python dict cache (simulating Redis hit/miss telemetry) and local ChromaDB + JSON metadata (skip PostgreSQL/Docker until after the hackathon demo)[cite: 1].
- **Frontend:** React + TypeScript + Tailwind CSS + Lucide React + Recharts (`frontend/`), styled in a **Swiss Technical Grid** aesthetic (`#F6F5F0` off-white background, `#111111` ink, `1px` `#D4D2C8` borders, `#FF4F00` signal-orange accents, `JetBrains Mono` metrics, zero dark-mode AI slop/glassmorphism).

## 3. Directory Map
- `backend/app/core/config.py` — Pydantic settings loading `.env` (`GEMINI_API_KEY`, `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`, `RRF_K=60`)
- `backend/app/schemas/` — Pydantic request/response models
- `backend/app/services/` — `hybrid_engine.py`, `agent_graph.py`, `web_search.py`, `evaluator.py`
- `backend/app/main.py` — FastAPI routes (`/health`, `/documents/upload`, `/documents`, `/query`, `/analytics`) with CORS enabled for `http://localhost:5173`[cite: 1]
- `backend/evaluation/` — Labeled Q&A dataset + script calculating `Recall@5`, `Precision@5`, `MRR`, and `P95 Latency` across Vector vs. BM25 vs. Hybrid RRF[cite: 1]
- `frontend/` — React + Tailwind 3-pane Retrieval Workbench & Evaluation Dashboard[cite: 1]

## 4. Mandatory Agent Rules
1. **Incremental Execution:** Build and test one sprint/module at a time. Never overwrite unrelated files.
2. **Verify Before Claiming Done:** Run Python syntax/import checks or `pytest` in `backend/venv` after creating files and inspect the output.
3. **Full Telemetry in `/query`:** Always return `rrf_score`, `bm25_rank`, `vector_rank`, `is_relevant`, `trace_steps`, `source_mode`, `web_sources`, `cache_hit`, and `latency_ms` so the frontend X-Ray Inspector has full audit data[cite: 1].
4. **Security & Git Hygiene:** Never read, print, or commit `.env` or `backend/data/`.