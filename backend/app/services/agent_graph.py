import json
import logging
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


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="Whether the chunk contains facts relevant to answering the query")
    explanation: str = Field(description="Brief explanation of why the chunk is or is not relevant")


def get_gemini_client():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=settings.GEMINI_API_KEY)


# Node 1: retrieve_kb
def retrieve_kb(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["question"]
    trace_steps = list(state.get("trace_steps", []))
    
    chunks = hybrid_engine.hybrid_search(query, top_k=5)
    step_msg = f"Retrieved {len(chunks)} chunks via Hybrid RRF for query: '{query}'"
    trace_steps.append(step_msg)
    
    return {
        "chunks": chunks,
        "trace_steps": trace_steps
    }


# Node 2: grade_relevance
def grade_relevance(state: AgentState) -> dict:
    question = state["question"]
    chunks = list(state.get("chunks", []))
    trace_steps = list(state.get("trace_steps", []))
    
    if not chunks:
        trace_steps.append("Graded relevance: No chunks to evaluate (0 relevant)")
        return {"chunks": [], "trace_steps": trace_steps}

    client = get_gemini_client()
    graded_chunks = []
    relevant_count = 0

    for chunk in chunks:
        prompt = f"""You are a strict retrieval evaluator.
Determine if the following context snippet contains information relevant to answering the question.

Question:
{question}

Context Snippet:
{chunk.text}
"""
        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RelevanceGrade,
                    temperature=0.0
                )
            )
            parsed = json.loads(response.text)
            is_rel = bool(parsed.get("is_relevant", False))
        except Exception as e:
            logger.warning(f"Grading failed for chunk {chunk.chunk_id}: {e}")
            is_rel = False

        chunk_dict = chunk.model_dump()
        chunk_dict["is_relevant"] = is_rel
        graded_chunk = ChunkResult(**chunk_dict)
        graded_chunks.append(graded_chunk)
        if is_rel:
            relevant_count += 1

    trace_steps.append(f"Graded relevance: {relevant_count}/{len(chunks)} chunks judged relevant")
    return {
        "chunks": graded_chunks,
        "trace_steps": trace_steps
    }


# Conditional Edge Router
def route_after_grading(state: AgentState) -> str:
    chunks = state.get("chunks", [])
    relevant_chunks = [c for c in chunks if c.is_relevant is True]
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
        response = client.models.generate_content(
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
If the context does not contain enough information to answer the question, state clearly what is missing.

Context:
{context_str}

User Question:
{question}

Answer:"""
        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            answer = response.text.strip()
            final_mode = "kb"
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            final_mode = "kb"

        trace_steps.append("Synthesized grounded answer from Knowledge Base with inline citations [filename · p.X]")

    elif web_sources:
        # Web Grounded Synthesis
        context_blocks = []
        for ws in web_sources:
            context_blocks.append(f"[{ws.title}]({ws.url})\n{ws.snippet}")
        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are VeriRAG, an auditable technical RAG assistant operating in Web Fallback mode.
Answer the user's question accurately using the live web search results below.
Cite every factual claim with markdown hyperlinks formatted exactly as: [Title](URL).

Web Search Results:
{context_str}

User Question:
{question}

Answer:"""
        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            answer = response.text.strip()
            final_mode = "web_fallback"
        except Exception as e:
            answer = f"Error generating web-fallback answer: {str(e)}"
            final_mode = "web_fallback"

        trace_steps.append("Synthesized web-grounded answer from Tavily search with markdown links [Title](URL)")

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
