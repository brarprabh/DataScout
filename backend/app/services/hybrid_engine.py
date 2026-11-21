import os
import re
import pymupdf as fitz
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.rag import ChunkResult


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridEngine:
    def __init__(self):
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="verirag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.gemini_client = None
        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

        self.chunks_store: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self._load_existing_chunks()

    def _get_embedding(self, text: str) -> List[float]:
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not configured.")
        response = self.gemini_client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text
        )
        # Handle embedding output structure
        if hasattr(response, "embedding"):
            return response.embedding.values
        elif hasattr(response, "embeddings") and response.embeddings:
            return response.embeddings[0].values
        raise ValueError("Could not extract embedding from Gemini response.")

    def _load_existing_chunks(self):
        """Loads all existing documents from ChromaDB to populate in-memory BM25 index."""
        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.chunks_store = []
            tokenized_corpus = []
            for i, chunk_id in enumerate(existing["ids"]):
                meta = existing["metadatas"][i] if existing.get("metadatas") else {}
                doc_text = existing["documents"][i] if existing.get("documents") else ""
                item = {
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 1)
                }
                self.chunks_store.append(item)
                tokenized_corpus.append(tokenize(doc_text))
            
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)

    def extract_and_chunk_pdf(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text page by page using PyMuPDF and split into 500-char chunks
        with 50-char sliding window overlap, preserving metadata.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        chunks = []
        chunk_idx = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text:
                continue

            cleaned_text = " ".join(text.split())
            if not cleaned_text:
                continue

            # Sliding window of CHUNK_SIZE characters with CHUNK_OVERLAP overlap
            step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
            for start in range(0, len(cleaned_text), step):
                chunk_text = cleaned_text[start : start + settings.CHUNK_SIZE]
                if not chunk_text.strip():
                    continue

                chunk_id = f"{filename}_p{page_num + 1}_c{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": filename,
                    "page": page_num + 1
                })
                chunk_idx += 1

        return chunks

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Synchronously indexes chunks into ChromaDB and rebuilds the BM25 index.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

        # Generate embeddings
        embeddings = [self._get_embedding(doc) for doc in documents]

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Update in-memory chunk store and BM25 index
        for c in chunks:
            # Replace if already exists, else append
            existing_idx = next((i for i, item in enumerate(self.chunks_store) if item["chunk_id"] == c["chunk_id"]), None)
            if existing_idx is not None:
                self.chunks_store[existing_idx] = c
            else:
                self.chunks_store.append(c)

        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks_store]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return len(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for c in self.chunks_store:
            src = c["source"]
            counts[src] = counts.get(src, 0) + 1
        return [{"filename": src, "chunk_count": count} for src, count in counts.items()]

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, int, float]]:
        """Returns list of (chunk_id, rank, score) sorted by BM25 score."""
        if not self.bm25 or not self.chunks_store:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_chunks = [
            (self.chunks_store[i]["chunk_id"], scores[i])
            for i in range(len(self.chunks_store))
            if scores[i] > 0
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [(cid, rank + 1, score) for rank, (cid, score) in enumerate(scored_chunks[:top_k])]

    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[str, int, float]]:
        """Returns list of (chunk_id, rank, distance) sorted by vector similarity."""
        if self.collection.count() == 0:
            return []

        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        ranked = []
        if results and results.get("ids") and results["ids"][0]:
            for rank, (cid, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
                ranked.append((cid, rank + 1, dist))
        return ranked

    def hybrid_search(self, query: str, top_k: int = 5) -> List[ChunkResult]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF(d) = sum(1 / (k + rank(d)))
        Combines BM25 and Vector rankings.
        """
        if not self.chunks_store:
            return []

        search_k = max(top_k * 2, 20)
        bm25_results = self.search_bm25(query, top_k=search_k)
        vector_results = self.search_vector(query, top_k=search_k)

        bm25_rank_map = {cid: rank for cid, rank, _ in bm25_results}
        vector_rank_map = {cid: rank for cid, rank, _ in vector_results}

        all_cids = set(bm25_rank_map.keys()) | set(vector_rank_map.keys())
        k = settings.RRF_K

        scored_results = []
        chunk_lookup = {c["chunk_id"]: c for c in self.chunks_store}

        for cid in all_cids:
            score = 0.0
            if cid in bm25_rank_map:
                score += 1.0 / (k + bm25_rank_map[cid])
            if cid in vector_rank_map:
                score += 1.0 / (k + vector_rank_map[cid])

            chunk_meta = chunk_lookup.get(cid)
            if not chunk_meta:
                continue

            scored_results.append({
                "chunk_id": cid,
                "text": chunk_meta["text"],
                "source": chunk_meta["source"],
                "page": chunk_meta["page"],
                "rrf_score": round(score, 6),
                "bm25_rank": bm25_rank_map.get(cid),
                "vector_rank": vector_rank_map.get(cid),
                "is_relevant": None
            })

        scored_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_chunks = scored_results[:top_k]
        return [ChunkResult(**item) for item in top_chunks]

    def generate_answer(self, query: str, chunks: List[ChunkResult]) -> str:
        """
        Synthesizes a grounded answer with inline citations [filename · p.X]
        using gemini-2.5-flash.
        """
        if not self.gemini_client:
            raise ValueError("GEMINI_API_KEY is not configured.")

        context_blocks = []
        for c in chunks:
            context_blocks.append(f"[{c.source} · p.{c.page}]\n{c.text}")
        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are VeriRAG, an auditable technical RAG assistant.
Answer the user's question accurately using ONLY the provided context snippets.
Cite the source for every factual statement using inline citations formatted exactly as: [filename · p.X].
If the context does not contain enough information to answer the question, state clearly what is missing.

Context:
{context_str}

User Question:
{query}

Answer:"""

        response = self.gemini_client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        return response.text.strip()


hybrid_engine = HybridEngine()
