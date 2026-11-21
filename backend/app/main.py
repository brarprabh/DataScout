import os
import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.rag import QueryRequest, QueryResponse, ChunkResult, DocumentInfo
from app.services.hybrid_engine import hybrid_engine
from app.services.agent_graph import agent_app

app = FastAPI(
    title="VeriRAG Engine",
    description="Auditable Hybrid RAG Engineering Workbench with RRF & Self-Correction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory query cache for hackathon latency & hit tracking
query_cache: Dict[str, Dict[str, Any]] = {}
cache_stats = {"hits": 0, "misses": 0}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "VeriRAG Engine",
        "gemini_configured": bool(settings.GEMINI_API_KEY)
    }


@app.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
        chunks = hybrid_engine.extract_and_chunk_pdf(content, file.filename)
        indexed_count = hybrid_engine.index_chunks(chunks)
        # Clear cache when knowledge base changes
        query_cache.clear()
        return {
            "message": f"Successfully indexed {file.filename}",
            "filename": file.filename,
            "chunks_indexed": indexed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
def get_documents():
    return hybrid_engine.list_documents()


@app.get("/analytics")
def get_analytics():
    benchmark_file = "backend/data/benchmark_results.json"
    benchmark_data = None
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception as e:
            benchmark_data = {"error": f"Failed to load benchmark data: {str(e)}"}

    total_chunks = len(hybrid_engine.chunks_store)
    documents = hybrid_engine.list_documents()
    total_docs = len(documents)

    total_cache_requests = cache_stats["hits"] + cache_stats["misses"]
    hit_rate = round(cache_stats["hits"] / total_cache_requests, 4) if total_cache_requests > 0 else 0.0

    return {
        "status": "success",
        "live_metrics": {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "cache_entries": len(query_cache),
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": hit_rate,
            "documents": documents
        },
        "benchmarks": benchmark_data
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.time()
    cache_key = request.query.strip().lower()

    if cache_key in query_cache:
        cache_stats["hits"] += 1
        cached = query_cache[cache_key]
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResponse(
            answer=cached["answer"],
            source_mode=cached["source_mode"],
            trace_steps=cached["trace_steps"] + ["cache_hit_retrieval"],
            chunks=cached["chunks"],
            web_sources=cached.get("web_sources", []),
            cache_hit=True,
            latency_ms=latency_ms
        )

    cache_stats["misses"] += 1

    initial_state = {
        "question": request.query,
        "rewritten_query": None,
        "retry_count": 0,
        "chunks": [],
        "web_sources": [],
        "source_mode": "none",
        "trace_steps": ["start_workflow"],
        "answer": ""
    }

    try:
        final_state = agent_app.invoke(initial_state)
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResponse(
            answer=f"Error executing agent pipeline: {str(e)}",
            source_mode="none",
            trace_steps=["start_workflow", f"error: {str(e)}"],
            chunks=[],
            web_sources=[],
            cache_hit=False,
            latency_ms=latency_ms
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    result_data = {
        "answer": final_state.get("answer", ""),
        "source_mode": final_state.get("source_mode", "none"),
        "trace_steps": final_state.get("trace_steps", []),
        "chunks": final_state.get("chunks", []),
        "web_sources": final_state.get("web_sources", []),
        "cache_hit": False,
        "latency_ms": latency_ms
    }

    query_cache[cache_key] = result_data
    return QueryResponse(**result_data)
