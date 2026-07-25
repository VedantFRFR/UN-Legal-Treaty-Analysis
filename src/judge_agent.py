import os
import json
import logging
from typing import Optional, List, Tuple, Dict, Any, TypedDict

from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-120b"

THRESHOLDS = {
    "definitional": {"strong": 0.0, "weak": 0.0, "min_strong": 0},
    "provision":    {"strong": 0.80, "weak": 0.60, "min_strong": 2},
    "conceptual":   {"strong": 0.55, "weak": 0.35, "min_strong": 3},
}


class JudgeState(TypedDict):
    query: str
    query_type: str
    chunks: List[Dict[str, Any]]
    doc_filter: Optional[str]
    article_filter: Optional[str]
    scored: List[Tuple[float, Dict[str, Any]]]
    strong: List[Tuple[float, Dict[str, Any]]]
    weak: List[Tuple[float, Dict[str, Any]]]
    re_retrieval_done: bool
    verdict: str


def _groq_client():
    key = os.environ.get("GROQ_FALLBACK_JUDGE_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No Groq API key found for judge agent. Set GROQ_FALLBACK_JUDGE_KEY or GROQ_API_KEY in .env")
    return Groq(api_key=key)


def _parse_scores(raw: List[Any], expected: int) -> List[float]:
    parsed = []
    for item in raw[:expected]:
        if isinstance(item, (int, float)):
            parsed.append(max(0.0, min(1.0, float(item))))
        elif isinstance(item, dict):
            parsed.append(max(0.0, min(1.0, float(item.get("score", 0.5)))))
        else:
            parsed.append(0.5)
    while len(parsed) < expected:
        parsed.append(0.5)
    return parsed


def _call_judge_llm(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    if not chunks:
        return []

    chunk_lines = []
    for i, c in enumerate(chunks):
        text = c.get("text", "")[:500]
        cid = c.get("id", "")
        chunk_lines.append(f"{i+1}. [{cid}]: {text}")

    prompt = f"""Given the query: "{query}"

Rate each chunk's relevance to answering the query on a scale of 0.0 to 1.0.

{chr(10).join(chunk_lines)}

Return ONLY valid JSON: {{"scores": [score1, score2, ...], "reasoning": "brief"}}"""

    try:
        client = _groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a legal relevance scorer. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        raw = data.get("scores", [])
        scores = _parse_scores(raw, len(chunks))
        return scores
    except Exception as e:
        logger.warning(f"Judge LLM call failed: {e}, using uniform 0.5 scores")
        return [0.5] * len(chunks)


def score_chunks(state: JudgeState) -> dict:
    chunks = state["chunks"]
    scores = _call_judge_llm(state["query"], chunks)
    scored = list(zip(scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)

    t = THRESHOLDS.get(state["query_type"], THRESHOLDS["conceptual"])
    strong = [(s, c) for s, c in scored if s >= t["strong"]]
    weak = [(s, c) for s, c in scored if t["weak"] <= s < t["strong"]]

    return {
        "scored": scored,
        "strong": strong,
        "weak": weak,
    }


def decide_verdict(state: JudgeState) -> dict:
    t = THRESHOLDS.get(state["query_type"], THRESHOLDS["conceptual"])
    strong_count = len(state["strong"])
    weak_count = len(state["weak"])

    if strong_count >= t["min_strong"]:
        return {"verdict": "accept"}

    if not state["re_retrieval_done"]:
        gap = t["min_strong"] - strong_count
        return {"verdict": "re-retrieve", "re_retrieval_done": True}

    if strong_count == 0 and weak_count == 0:
        return {"verdict": "re-classify"}

    return {"verdict": "accept"}


def build_judge_graph():
    builder = StateGraph(JudgeState)

    builder.add_node("score_chunks", score_chunks)
    builder.add_node("decide_verdict", decide_verdict)

    builder.set_entry_point("score_chunks")
    builder.add_edge("score_chunks", "decide_verdict")
    builder.add_conditional_edges(
        "decide_verdict",
        lambda s: s["verdict"],
        {
            "accept": END,
            "re-retrieve": END,
            "re-classify": END,
        },
    )

    return builder.compile()


_judge_graph = None

def get_judge_graph():
    global _judge_graph
    if _judge_graph is None:
        _judge_graph = build_judge_graph()
    return _judge_graph


def run_judge(
    query: str,
    query_type: str,
    chunks: List[Dict[str, Any]],
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> Dict[str, Any]:
    graph = get_judge_graph()

    initial = JudgeState(
        query=query,
        query_type=query_type,
        chunks=chunks,
        doc_filter=doc_filter,
        article_filter=article_filter,
        scored=[],
        strong=[],
        weak=[],
        re_retrieval_done=False,
        verdict="accept",
    )

    result = graph.invoke(initial)
    return result
