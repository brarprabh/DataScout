from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int
    rrf_score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    is_relevant: Optional[bool] = None


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    source_mode: str
    trace_steps: List[str]
    chunks: List[ChunkResult]
    web_sources: List[WebSource]
    cache_hit: bool
    latency_ms: float


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
