import time
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.schemas.rag import QueryRequest, QueryResponse, ChunkResult, DocumentInfo
from app.services.hybrid_engine import hybrid_engine

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


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.time()
    cache_key = request.query.strip().lower()

    if cache_key in query_cache:
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

    trace_steps = ["start", "retrieve_hybrid_rrf"]
    chunks: List[ChunkResult] = hybrid_engine.hybrid_search(request.query, top_k=request.top_k)

    if not chunks:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return QueryResponse(
            answer="No relevant documents or chunks found in the knowledge base. Please upload a PDF.",
            source_mode="none",
            trace_steps=trace_steps + ["no_results"],
            chunks=[],
            web_sources=[],
            cache_hit=False,
            latency_ms=latency_ms
        )

    trace_steps.append("generate_grounded_answer")
    try:
        answer = hybrid_engine.generate_answer(request.query, chunks)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    trace_steps.append("completed")
    latency_ms = round((time.time() - start_time) * 1000, 2)

    result_data = {
        "answer": answer,
        "source_mode": "hybrid_kb",
        "trace_steps": trace_steps,
        "chunks": chunks,
        "web_sources": [],
        "cache_hit": False,
        "latency_ms": latency_ms
    }

    query_cache[cache_key] = result_data

    return QueryResponse(**result_data)
