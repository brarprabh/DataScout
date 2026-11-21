import json
import logging
import time
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.rag import ChunkResult, WebSource
from app.services.hybrid_engine import hybrid_engine
from app.services.web_search import web_search_service

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    rewritten_query: Optional[str]
    retry_count: int
    chunks: List[ChunkResult]
    web_sources: List[WebSource]
    source_mode: str  # "kb", "web_fallback", "none"
    trace_steps: List[str]
    answer: str


class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="1-based index of the snippet")
    is_relevant: bool = Field(description="Whether the snippet contains facts relevant to answering the question")


class BatchRelevanceGrade(BaseModel):
    grades: List[ChunkGrade] = Field(description="List of relevance judgements for each evaluated snippet")


TRANSIENT_ERRORS = (
    "503",
    "UNAVAILABLE",
    "high demand",
    "10053",
    "10054",
    "connection was aborted",
    "connection reset",
    "RemoteDisconnected",
    "timeout",
    "timed out"
)


def generate_content_with_retry(client, model: str, contents: str, config=None, max_retries: int = 4):
    """Executes client.models.generate_content with exponential backoff on transient network or 503 errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            err_msg = str(e).lower()
            is_transient = any(trigger.lower() in err_msg for trigger in TRANSIENT_ERRORS)
            if is_transient and attempt < max_retries - 1:
                wait_sec = (attempt + 1) * 2
                logger.warning(f"Transient error ({type(e).__name__}: {str(e)[:80]}). Retrying in {wait_sec}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_sec)
            else:
                raise e


def get_gemini_client():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=60.0)
    )


# Node 1: retrieve_kb (Optimized to top 3 chunks for high-speed evaluation)
def retrieve_kb(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    trace_steps = list(state.get("trace_steps", []))
    
    # Prune to top 3 chunks for fast RRF retrieval & minimal latency
    chunks = hybrid_engine.hybrid_search(query, top_k=3)
    step_msg = f"Retrieved {len(chunks)} chunks via Hybrid RRF for query: '{query}'"
    trace_steps.append(step_msg)
    
    return {
        "chunks": chunks,
        "trace_steps": trace_steps
    }


# Node 2: grade_relevance (Single-Prompt Batch Grading)
def grade_relevance(state: AgentState) -> dict:
    question = state["question"]
    chunks = list(state.get("chunks", []))
    trace_steps = list(state.get("trace_steps", []))
    
    if not chunks:
        trace_steps.append("Graded relevance: No chunks to evaluate (0 relevant)")
        return {"chunks": [], "trace_steps": trace_steps}

    client = get_gemini_client()

    # Build single consolidated evaluation prompt for all chunks
    snippets_text = "\n\n".join(
        [f"[Snippet {i + 1} - ID: {c.chunk_id}]\n{c.text}" for i, c in enumerate(chunks)]
    )

    prompt = f"""You are an auditable technical retrieval grader.
Evaluate whether each of the following context snippets contains information relevant to answering or explaining any aspect of the user's question.
Be generous: if a snippet mentions related topics, domain concepts, data fields, entities, or background information that helps answer the question, mark it as true. Only mark as false if it is completely unrelated.

User Question:
{question}

Context Snippets:
{snippets_text}

For each snippet from 1 to {len(chunks)}, determine if it is relevant (true) or completely irrelevant (false)."""

    grade_map = {}
    grading_succeeded = False
    try:
        response = generate_content_with_retry(
            client=client,
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchRelevanceGrade,
                temperature=0.0
            )
        )
        parsed = json.loads(response.text)
        for item in parsed.get("grades", []):
            idx = item.get("chunk_index")
            if idx is not None and 1 <= idx <= len(chunks):
                grade_map[idx - 1] = bool(item.get("is_relevant", False))
        grading_succeeded = True
    except Exception as e:
        logger.warning(f"Batch grading fallback triggered: {e}")
        grading_succeeded = False

    graded_chunks = []
    relevant_count = 0
    for i, chunk in enumerate(chunks):
        # Strict gate: Only mark relevant if the LLM explicitly judged it as True
        # If grading failed/timed out, mark as False (do not assume relevance without evidence)
        is_rel = grade_map.get(i, False) if grading_succeeded else False

        chunk_dict = chunk.model_dump()
        chunk_dict["is_relevant"] = is_rel
        graded_chunks.append(ChunkResult(**chunk_dict))
        if is_rel:
            relevant_count += 1

    trace_steps.append(f"Graded relevance: {relevant_count}/{len(chunks)} chunks judged relevant (batch evaluated)")
    return {
        "chunks": graded_chunks,
        "trace_steps": trace_steps
    }


# Conditional Edge Router
def route_after_grading(state: AgentState) -> str:
    chunks = state.get("chunks", [])
    # Confidence gating: Only consider chunks that are graded relevant AND have legitimate dual-retriever or high RRF confidence (> 65% match)
    # A chunk with RRF <= 0.016393 (50%) only matched vector cosine noise with 0 BM25 hits
    relevant_chunks = [
        c for c in chunks 
        if c.is_relevant is True and (c.bm25_rank is not None or c.rrf_score > 0.020)
    ]
    retry_count = state.get("retry_count", 0)

    if relevant_chunks:
        return "generate_answer_kb"
    elif retry_count == 0:
        return "rewrite_query"
    else:
        return "web_fallback"


# Node 3: rewrite_query
def rewrite_query(state: AgentState) -> dict:
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    trace_steps = list(state.get("trace_steps", []))
    
    client = get_gemini_client()
    prompt = f"""You are a query rewriting assistant for technical document retrieval.
The original query did not retrieve relevant passages from the technical knowledge base.
Formulate an improved search query using technical keywords, domain terminology, and error identifiers to maximize retrieval recall.
Return ONLY the rewritten search query with no additional commentary or punctuation.

Original Query:
{question}

Rewritten Query:"""

    try:
        response = generate_content_with_retry(
            client=client,
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        rewritten = response.text.strip()
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
        rewritten = question

    trace_steps.append(f"Rewrote query for technical recall: '{rewritten}' (retry_count={retry_count})")
    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count,
        "trace_steps": trace_steps
    }


# Node 4: web_fallback
def web_fallback(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    trace_steps = list(state.get("trace_steps", []))
    trace_steps.append(f"KB relevance weak; triggered Tavily web fallback for '{query}'")

    web_results = web_search_service.search(query=query, max_results=4)
    trace_steps.append(f"Retrieved {len(web_results)} live web sources via Tavily")

    return {
        "web_sources": web_results,
        "source_mode": "web_fallback",
        "trace_steps": trace_steps
    }


# Node 5: generate_answer
def generate_answer(state: AgentState) -> dict:
    question = state["question"]
    source_mode = state.get("source_mode", "kb")
    chunks = state.get("chunks", [])
    web_sources = state.get("web_sources", [])
    trace_steps = list(state.get("trace_steps", []))
    client = get_gemini_client()

    relevant_chunks = [c for c in chunks if c.is_relevant is True]
    if not relevant_chunks and chunks and source_mode == "kb":
        # Fallback to top RRF chunks if grading marked none
        relevant_chunks = chunks[:3]

    if source_mode != "web_fallback" and relevant_chunks:
        # Knowledge Base Grounded Synthesis
        context_blocks = []
        for c in relevant_chunks:
            context_blocks.append(f"[{c.source} · p.{c.page}]\n{c.text}")
        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are VeriRAG, an auditable technical RAG assistant.
Answer the user's question accurately using ONLY the provided context snippets.
Cite the source for every factual statement using inline citations formatted exactly as: [filename · p.X].

Structure your answer with:
1. Direct Explanation / Core Findings (with inline citations [filename · p.X])
2. Key Points / Technical Details
3. Conclusion / Executive Summary (synthesizing the key takeaways clearly)

If the context does not contain enough information to answer the question, state clearly what is missing.

Context:
{context_str}

User Question:
{question}

Answer:"""
        try:
            response = generate_content_with_retry(
                client=client,
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            answer = response.text.strip()
            final_mode = "kb"
            trace_steps.append("Synthesized grounded answer from Knowledge Base with inline citations [filename · p.X]")
        except Exception as e:
            logger.warning(f"KB synthesis encountered error/timeout: {e}. Falling back to structured extractive answer.")
            # Resilient Extractive Fallback: deliver the relevant knowledge base chunks directly
            lines = [
                f"**Extracted Knowledge Base Findings for:** *{question}*\n",
                f"*(Note: LLM synthesis encountered a transient network timeout; displaying verified excerpts directly from your document)*\n"
            ]
            for c in relevant_chunks:
                lines.append(f"**From [{c.source} · p.{c.page}]:**\n> {c.text.strip()}\n")
            answer = "\n".join(lines)
            final_mode = "kb"
            trace_steps.append("Directly presented relevant Knowledge Base passages with [filename · p.X] citations")

    elif web_sources:
        # Web Grounded Synthesis
        context_blocks = []
        for ws in web_sources:
            context_blocks.append(f"[{ws.title}]({ws.url})\n{ws.snippet}")
        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are VeriRAG, an auditable technical RAG assistant operating in Web Fallback mode.
Answer the user's question accurately using the live web search results below.
Cite every factual claim with markdown hyperlinks formatted exactly as: [Title](URL).

Structure your answer with:
1. Direct Answer & Verified Facts (citing [Title](URL))
2. Supporting Context & Details
3. Executive Summary / Conclusion

Web Search Results:
{context_str}

User Question:
{question}

Answer:"""
        try:
            response = generate_content_with_retry(
                client=client,
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            answer = response.text.strip()
            final_mode = "web_fallback"
            trace_steps.append("Synthesized web-grounded answer from Tavily search with markdown links [Title](URL)")
        except Exception as e:
            logger.warning(f"Gemini web synthesis error, falling back to direct extractive web summary: {e}")
            # Resilient fallback: build structured web summary directly from retrieved Tavily sources
            summary_lines = [
                f"**Web Search Results for:** *{question}*\n",
                f"*(Note: LLM synthesis encountered a transient network delay; showing direct verified extracts from Tavily)*\n"
            ]
            for ws in web_sources:
                summary_lines.append(f"- **[{ws.title}]({ws.url})**\n  {ws.snippet}\n")
            answer = "\n".join(summary_lines)
            final_mode = "web_fallback"
            trace_steps.append("Directly formatted Tavily web search sources into structured summary")

    else:
        answer = "No relevant information found in the internal knowledge base or web search."
        final_mode = "none"
        trace_steps.append("No context available to generate answer")

    return {
        "answer": answer,
        "source_mode": final_mode,
        "trace_steps": trace_steps
    }


# Helper for routing to generate_answer with source_mode="kb"
def prepare_kb_answer(state: AgentState) -> dict:
    return {"source_mode": "kb"}


# Build LangGraph workflow
def build_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve_kb", retrieve_kb)
    builder.add_node("grade_relevance", grade_relevance)
    builder.add_node("prepare_kb_answer", prepare_kb_answer)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("web_fallback", web_fallback)
    builder.add_node("generate_answer", generate_answer)

    builder.set_entry_point("retrieve_kb")
    builder.add_edge("retrieve_kb", "grade_relevance")

    builder.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {
            "generate_answer_kb": "prepare_kb_answer",
            "rewrite_query": "rewrite_query",
            "web_fallback": "web_fallback"
        }
    )

    builder.add_edge("prepare_kb_answer", "generate_answer")
    builder.add_edge("rewrite_query", "retrieve_kb")
    builder.add_edge("web_fallback", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


agent_app = build_agent_graph()
