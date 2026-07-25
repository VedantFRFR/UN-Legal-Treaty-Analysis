import os
import json
import sys
import logging
from typing import Optional, List, Tuple, Dict, Any, TypedDict, Literal

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END
from src.judge_agent import run_judge, THRESHOLDS

from src.retrieval import (
    definitional_retrieval,
    provision_retrieval,
    conceptual_retrieval,
    fused_retrieval,
    rerank_chunks,
    diversity_filter,
    fetch_parents,
    fetch_siblings,
    assemble_context,
    compute_retrieval_confidence,
)

RETRIEVAL_STRATS = {
    "definitional": definitional_retrieval,
    "provision": provision_retrieval,
    "conceptual": conceptual_retrieval,
}

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-120b"


class RetrievalState(TypedDict):
    original_query: str
    rewritten_query: str
    doc_filter: Optional[str]
    article_filter: Optional[str]
    query_type: str
    confidence: float
    chunks: List[Dict[str, Any]]
    reranked_chunks: List[Tuple[float, Dict[str, Any]]]
    parent_chunks: Dict[str, Dict[str, Any]]
    sibling_info: Optional[Dict[str, List[Dict[str, Any]]]]
    context: str
    answer: str
    retrieval_confidence: float
    judge_verdict: str
    judge_reclassify_count: int
    chunks_retrieved: int
    chunks_post_filter: int
    error: Optional[str]


def _make_initial_state(query: str) -> RetrievalState:
    return RetrievalState(
        original_query=query,
        rewritten_query="",
        doc_filter=None,
        article_filter=None,
        query_type="conceptual",
        confidence=0.0,
        chunks=[],
        reranked_chunks=[],
        parent_chunks={},
        sibling_info=None,
        context="",
        answer="",
        retrieval_confidence=0.0,
        judge_verdict="accept",
        judge_reclassify_count=0,
        chunks_retrieved=0,
        chunks_post_filter=0,
        error=None,
    )


def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in .env or environment variables."
        )
    return Groq(api_key=api_key)


def _normalize_doc_filter(doc_val: str) -> Optional[str]:
    if not doc_val:
        return None
    doc_val = doc_val.strip()
    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if os.path.isfile(os.path.join(pdf_dir, doc_val)):
        return doc_val
    if not doc_val.lower().endswith(".pdf"):
        with_ext = doc_val + ".pdf"
        if os.path.isfile(os.path.join(pdf_dir, with_ext)):
            return with_ext
    return None


def _normalize_article_filter(art_val: str) -> Optional[str]:
    if not art_val:
        return None
    art_val = art_val.strip().removeprefix("Article ").removeprefix("Art. ").removeprefix("article ").removeprefix("art. ")
    if not art_val:
        return None
    if not all(c.isdigit() or c in "/.-" for c in art_val):
        return None
    return art_val


def rewrite_query(state: RetrievalState) -> dict:
    client = _get_groq_client()

    system_prompt = """You are a legal query rewriter for UN treaty research. Your task:
1. Expand vague terms to precise legal terminology (e.g., "exceptions" -> "exceptions derogations limitations exclusions")
2. Preserve exact article and document references verbatim
3. Keep the rewritten query under 50 words
4. Identify mentioned documents ONLY if the user explicitly names a document (e.g. "CEDAW", "ICCPR", "ICESCR", "Convention Against Torture", "Refugee Convention"). Do NOT guess or invent a document name.
5. Identify mentioned_article ONLY if the user explicitly cites a single article number (e.g. "Article 7", "Art. 18"). Do NOT infer or guess article numbers from context. Return a SINGLE article number or null. NEVER return multiple articles — if the query references multiple concepts without a single specific article, set mentioned_article to null.

Output valid JSON with these exact fields:
{
  "rewritten_query": "expanded legal query under 50 words",
  "mentioned_doc": "PDF filename like CEDAW.pdf or null if not explicitly mentioned",
  "mentioned_article": "single article number like '7' or null"
}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["original_query"]},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        rewritten = result.get("rewritten_query", state["original_query"])
        doc_filter = _normalize_doc_filter(result.get("mentioned_doc"))
        article_filter = _normalize_article_filter(result.get("mentioned_article"))
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        rewritten = state["original_query"]
        doc_filter = None
        article_filter = None

    return {
        "rewritten_query": rewritten,
        "doc_filter": doc_filter,
        "article_filter": article_filter,
    }


def classify_query(state: RetrievalState) -> dict:
    client = _get_groq_client()

    system_prompt = """You are a legal query classifier for UN treaty research.
Classify the query into exactly one type. Return ONLY valid JSON.

Types:
- definitional: looks up a specific article or definition
- provision: enumerates, lists, or retrieves multiple specific provisions, rights, or exceptions across an act
- conceptual: analyzes, compares, or synthesizes concepts across multiple articles

Few-shot examples:
"What does Article 7 say" -> definitional
"List all derogations in CAT" -> provision
"List all rights granted to refugees under the Refugee Convention" -> provision
"How are minority rights protected under international law" -> conceptual
"Compare protections for people under ICCPR and CATs" -> conceptual

Output:
{{
  "query_type": "definitional" | "provision" | "conceptual",
  "confidence": 0.0-1.0
}}

Query: {query}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.format(
                    query=state["original_query"]
                )},
                {"role": "user", "content": state["original_query"]},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        query_type = result.get("query_type", "conceptual")
        confidence = float(result.get("confidence", 0.0))
    except Exception as e:
        logger.warning(f"Query classification failed: {e}")
        query_type = "conceptual"
        confidence = 0.0

    if confidence < 0.70:
        query_type = "conceptual"

    return {"query_type": query_type, "confidence": confidence}


def route_after_judge(state: RetrievalState) -> str:
    if state.get("judge_verdict") == "re-classify":
        return "re-classify"
    return "continue"


def retrieve(state: RetrievalState) -> dict:
    strat = RETRIEVAL_STRATS.get(state["query_type"], conceptual_retrieval)
    chunks = strat(state["rewritten_query"], state["doc_filter"], state["article_filter"])
    return {"chunks": chunks, "chunks_retrieved": len(chunks)}


def judge_chunks(state: RetrievalState) -> dict:
    query = state["rewritten_query"]
    query_type = state["query_type"]
    chunks = state["chunks"]
    doc_f = state["doc_filter"]
    art_f = state["article_filter"]
    reclassify_count = state.get("judge_reclassify_count", 0)

    print(f"  [Judge Agent] Scoring {len(chunks)} chunks for '{query_type}' query...")

    result = run_judge(query, query_type, chunks, doc_f, art_f)
    strong = result.get("strong", [])
    weak = result.get("weak", [])
    verdict = result["verdict"]

    print(f"  [Judge Agent] Strong: {len(strong)}, Weak: {len(weak)}, Verdict: {verdict}")

    if verdict == "re-retrieve":
        t = THRESHOLDS.get(query_type, THRESHOLDS["conceptual"])
        gap = max(0, t["min_strong"] - len(strong))
        print(f"  [Judge Agent] Re-retrieving {gap * 2} more chunks...")
        extra = fused_retrieval(
            query, top_k=gap * 2, similarity_threshold=0.05,
            doc_filter=doc_f, article_filter=art_f,
        )
        all_chunks = chunks + extra
        result = run_judge(query, query_type, all_chunks, doc_f, art_f)
        strong = result.get("strong", [])
        weak = result.get("weak", [])
        verdict = result["verdict"]
        print(f"  [Judge Agent] After re-retrieval: Strong: {len(strong)}, Weak: {len(weak)}, Verdict: {verdict}")

    if verdict == "re-classify" and reclassify_count == 0:
        print(f"  [Judge Agent] Re-classifying (no strong chunks found)...")
        promoted = [(s * 0.9, c) for s, c in weak]
        all_scored = strong + promoted
        all_scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "reranked_chunks": all_scored,
            "judge_verdict": "re-classify",
            "judge_reclassify_count": 1,
        }

    all_scored = strong + weak
    all_scored.sort(key=lambda x: x[0], reverse=True)

    if not all_scored and chunks:
        print(f"  [Judge Agent] All chunks rejected! Fallback: keeping top 3.")
        top3 = chunks[:3]
        all_scored = [(0.4, c) for c in top3]

    print(f"  [Judge Agent] Accepting {len(all_scored)} chunks for generation.")
    return {
        "reranked_chunks": all_scored,
        "judge_verdict": "accept",
    }


def rerank_and_filter(state: RetrievalState) -> dict:
    if state["query_type"] == "definitional" or state.get("article_filter"):
        reranked = [(1.0, c) for c in state["chunks"]]
        filtered = [(s, c) for s, c in reranked]
    elif state.get("judge_verdict") == "accept":
        reranked = state.get("reranked_chunks") or [(1.0, c) for c in state["chunks"]]
        threshold, min_chunks = (
            (0.002, 5) if state["query_type"] == "provision" else (0.02, 3)
        )
        filtered = diversity_filter(reranked, max_per_article=3)
        if len(filtered) < min_chunks:
            seen_ids = {c.get("id") for _, c in filtered}
            for score, chunk in reranked:
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                filtered.append((score, chunk))
                seen_ids.add(cid)
                if len(filtered) >= min_chunks:
                    break
    else:
        threshold, min_chunks = (
            (0.002, 5) if state["query_type"] == "provision" else (0.02, 3)
        )
        reranked = rerank_chunks(
            state["rewritten_query"],
            state["chunks"],
            threshold=threshold,
            min_chunks=min_chunks,
        )
        filtered = diversity_filter(reranked, max_per_article=3)
        if len(filtered) < min_chunks:
            seen_ids = {c.get("id") for _, c in filtered}
            for score, chunk in reranked:
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                filtered.append((score, chunk))
                seen_ids.add(cid)
                if len(filtered) >= min_chunks:
                    break

    return {
        "reranked_chunks": filtered,
        "chunks_post_filter": len(filtered),
    }


def traverse(state: RetrievalState) -> dict:
    if state["query_type"] == "definitional":
        return {"parent_chunks": {}, "sibling_info": None}

    parents = fetch_parents(state["reranked_chunks"])

    sibling_info = None
    if state["query_type"] == "provision":
        sibling_info = fetch_siblings(state["reranked_chunks"], top_n=5)

    return {"parent_chunks": parents, "sibling_info": sibling_info}


def assemble(state: RetrievalState) -> dict:
    context = assemble_context(
        query_type=state["query_type"],
        reranked_chunks=state["reranked_chunks"],
        parent_chunks=state["parent_chunks"],
        sibling_info=state["sibling_info"],
        article_filter=state.get("article_filter"),
    )
    return {"context": context}


def generate_answer(state: RetrievalState) -> dict:
    client = _get_groq_client()

    rc = compute_retrieval_confidence(state["reranked_chunks"])

    system_prompt = """You are a legal analysis assistant for UN treaty research.
Answer the user's query based on the provided context.

Guidelines:
- Add inline citations only when you directly use content from a chunk
- Do not cite all retrieved chunks — only those actually used
- Citation format: [DOC_NAME, Article X, Clause Y]
- Be precise and concise
- If the context is limited, answer based on what is available and briefly note the scope"""

    user_prompt = f"""Query: {state["original_query"]}

Context:
{state["context"]}

Answer the query with appropriate inline citations."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "Failed to generate an answer due to an error."

    return {
        "answer": answer,
        "retrieval_confidence": rc,
    }


def finalize(state: RetrievalState) -> dict:
    return {}


def build_retrieval_graph():
    from src.retrieval import _embed_query
    _embed_query("init")

    builder = StateGraph(RetrievalState)

    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("judge_chunks", judge_chunks)
    builder.add_node("rerank_and_filter", rerank_and_filter)
    builder.add_node("traverse", traverse)
    builder.add_node("assemble", assemble)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("rewrite_query")

    builder.add_edge("rewrite_query", "classify_query")
    builder.add_edge("classify_query", "retrieve")
    builder.add_edge("retrieve", "judge_chunks")

    builder.add_conditional_edges(
        "judge_chunks",
        route_after_judge,
        {
            "re-classify": "classify_query",
            "continue": "rerank_and_filter",
        },
    )

    builder.add_edge("rerank_and_filter", "traverse")
    builder.add_edge("traverse", "assemble")
    builder.add_edge("assemble", "generate_answer")
    builder.add_edge("generate_answer", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


_retrieval_graph = None


def get_retrieval_graph():
    global _retrieval_graph
    if _retrieval_graph is None:
        _retrieval_graph = build_retrieval_graph()
    return _retrieval_graph


def run_pipeline(query: str) -> Dict[str, Any]:
    graph = get_retrieval_graph()
    initial = _make_initial_state(query)
    result = graph.invoke(initial)

    return {
        "answer": result.get("answer", ""),
        "query_type": result.get("query_type", "conceptual"),
        "retrieval_confidence": result.get("retrieval_confidence", 0.0),
        "chunks_retrieved": result.get("chunks_retrieved", 0),
        "chunks_post_filter": result.get("chunks_post_filter", 0),
        "error": result.get("error"),
    }
