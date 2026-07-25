import os
import re
import sys
from typing import Optional, List, Tuple, Dict, Any, Set
from collections import defaultdict

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from settings import CHROMA_DIR, CHROMA_COLLECTION


def _get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection(client):
    return client.get_collection(name=CHROMA_COLLECTION)


def _l2_to_similarity(l2_dist: float) -> float:
    return 1.0 - (l2_dist ** 2) / 2.0


def _similarity_to_l2(similarity: float) -> float:
    return (2.0 * (1.0 - similarity)) ** 0.5


_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        from settings import EMBEDDING_MODEL, EMBEDDING_MAX_LENGTH
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        if EMBEDDING_MAX_LENGTH > 0:
            _embed_model.max_seq_length = EMBEDDING_MAX_LENGTH
    return _embed_model


def _embed_query(text: str) -> List[float]:
    model = _get_embed_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


_bm25_index = None
_bm25_docs: List[Dict] = []
_bm25_id_to_idx: Dict[str, int] = {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _build_bm25_index():
    global _bm25_index, _bm25_docs, _bm25_id_to_idx

    if _bm25_index is not None:
        return

    client = _get_client()
    collection = _get_collection(client)

    all_data = collection.get(include=["documents", "metadatas"])

    if not all_data["ids"]:
        _bm25_index = None
        return

    _bm25_docs = []
    _bm25_id_to_idx = {}
    corpus = []

    for i, doc_id in enumerate(all_data["ids"]):
        text = all_data["documents"][i] or ""
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        _bm25_docs.append({"id": doc_id, "text": text, "metadata": meta})
        _bm25_id_to_idx[doc_id] = i
        corpus.append(_tokenize(text))

    from rank_bm25 import BM25Okapi
    _bm25_index = BM25Okapi(corpus)

    print(f"  BM25 index built: {len(corpus)} documents")


def rebuild_bm25():
    global _bm25_index, _bm25_docs, _bm25_id_to_idx
    _bm25_index = None
    _bm25_docs = []
    _bm25_id_to_idx = {}
    _build_bm25_index()
    print("  BM25 index rebuilt.")


def _bm25_query(
    query_text: str,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _build_bm25_index()

    if _bm25_index is None or not _bm25_docs:
        return []

    tokenized_query = _tokenize(query_text)
    raw_scores = _bm25_index.get_scores(tokenized_query)

    scored: List[Tuple[float, int]] = []
    for idx in range(len(_bm25_docs)):
        doc = _bm25_docs[idx]
        meta = doc["metadata"]

        if doc_filter and meta.get("doc_name") != doc_filter:
            continue
        if article_filter and meta.get("article_no") != article_filter and meta.get("parent_article_no") != article_filter:
            continue

        scored.append((raw_scores[idx], idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    results = []
    for score, idx in top:
        doc = _bm25_docs[idx]
        chunk = dict(doc["metadata"])
        chunk["text"] = doc["text"]
        chunk["id"] = doc["id"]
        chunk["bm25_score"] = round(float(score), 4)
        results.append(chunk)

    return results


_RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    rrf_threshold: float = 0.0,
) -> List[Tuple[float, Dict[str, Any]]]:
    vector_ranks: Dict[str, int] = {
        c["id"]: idx + 1 for idx, c in enumerate(vector_results)
    }
    bm25_ranks: Dict[str, int] = {
        c["id"]: idx + 1 for idx, c in enumerate(bm25_results)
    }

    all_ids: Set[str] = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    rrf_scores: Dict[str, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in vector_ranks:
            score += 1.0 / (_RRF_K + vector_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (_RRF_K + bm25_ranks[doc_id])
        rrf_scores[doc_id] = round(score, 6)

    filtered = {
        doc_id: score
        for doc_id, score in rrf_scores.items()
        if score >= rrf_threshold
    }

    sorted_ids = sorted(filtered.keys(), key=lambda x: filtered[x], reverse=True)

    chunk_lookup: Dict[str, Dict[str, Any]] = {}
    for c in vector_results:
        chunk_lookup[c["id"]] = c
    for c in bm25_results:
        if c["id"] not in chunk_lookup:
            chunk_lookup[c["id"]] = c

    fused: List[Tuple[float, Dict[str, Any]]] = []
    for doc_id in sorted_ids:
        chunk = dict(chunk_lookup[doc_id])
        chunk["rrf_score"] = filtered[doc_id]
        fused.append((filtered[doc_id], chunk))

    return fused


def fused_retrieval(
    query_text: str,
    top_k: int = 10,
    rrf_threshold: float = 0.01,
    similarity_threshold: Optional[float] = None,
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    embedding = _embed_query(query_text)

    client = _get_client()
    collection = _get_collection(client)

    filters = []
    if doc_filter:
        filters.append({"doc_name": doc_filter})
    if article_filter:
        filters.append({"$or": [{"article_no": article_filter}, {"parent_article_no": article_filter}]})
    where_clause = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)

    vector_results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 3,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    vector_chunks: List[Dict[str, Any]] = []
    if vector_results["ids"] and vector_results["ids"][0]:
        for i, doc_id in enumerate(vector_results["ids"][0]):
            l2_dist = vector_results["distances"][0][i]
            sim = _l2_to_similarity(l2_dist)
            meta = vector_results["metadatas"][0][i] if vector_results["metadatas"] else {}
            chunk = dict(meta)
            chunk["text"] = vector_results["documents"][0][i]
            chunk["id"] = doc_id
            chunk["similarity"] = round(sim, 4)
            vector_chunks.append(chunk)
    if similarity_threshold is not None:
        vector_chunks = [
            c for c in vector_chunks
            if c.get("similarity", 0) >= similarity_threshold
        ]
    vector_chunks = vector_chunks[:top_k]

    bm25_chunks = _bm25_query(
        query_text=query_text,
        top_k=top_k,
        doc_filter=doc_filter,
        article_filter=article_filter,
    )

    fused = reciprocal_rank_fusion(
        vector_chunks,
        bm25_chunks,
        rrf_threshold=rrf_threshold,
    )

    return [chunk for _, chunk in fused]


def fetch_chunks_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}
    client = _get_client()
    collection = _get_collection(client)
    results = collection.get(
        ids=ids,
        include=["documents", "metadatas"],
    )
    out = {}
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results["metadatas"] else {}
        chunk = dict(meta)
        chunk["text"] = results["documents"][i]
        chunk["id"] = doc_id
        out[doc_id] = chunk
    return out


def _path_up(path: str, levels: int = 1) -> Optional[str]:
    parts = path.split("/")
    if len(parts) <= levels:
        return None
    return "/".join(parts[:-levels])


def _infer_depth(chunk: Dict[str, Any]) -> int:
    d = chunk.get("depth")
    if d is not None:
        return int(d)
    ct = chunk.get("chunk_type")
    if ct == "article":
        return 0
    if ct == "clause":
        return 1
    if ct == "subclause":
        return 2
    return -1


def definitional_retrieval(
    rewritten_query: str,
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return fused_retrieval(
        query_text=rewritten_query,
        top_k=5,
        rrf_threshold=0.015,
        similarity_threshold=0.15,
        doc_filter=doc_filter,
        article_filter=article_filter,
    )


def provision_retrieval(
    rewritten_query: str,
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    chunks = fused_retrieval(
        query_text=rewritten_query,
        top_k=30,
        rrf_threshold=0.015,
        similarity_threshold=0.10,
        doc_filter=doc_filter,
        article_filter=article_filter,
    )
    if len(chunks) > 30:
        chunks = fused_retrieval(
            query_text=rewritten_query,
            top_k=30,
            rrf_threshold=0.015,
            similarity_threshold=0.20,
            doc_filter=doc_filter,
            article_filter=article_filter,
        )
    return chunks


def conceptual_retrieval(
    rewritten_query: str,
    doc_filter: Optional[str] = None,
    article_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return fused_retrieval(
        query_text=rewritten_query,
        top_k=20,
        rrf_threshold=0.015,
        similarity_threshold=0.08,
        doc_filter=doc_filter,
        article_filter=article_filter,
    )


_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        import torch
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", activation_fn=torch.sigmoid)
    return _reranker_model


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    threshold: float = 0.002,
    min_chunks: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(scores.tolist(), chunks), key=lambda x: x[0], reverse=True)

    filtered = [(s, c) for s, c in reranked if s > threshold]
    if len(filtered) < min_chunks:
        remaining = [(s, c) for s, c in reranked if (s, c) not in filtered]
        needed = min_chunks - len(filtered)
        filtered.extend(remaining[:needed])

    return filtered


def diversity_filter(
    chunks: List[Tuple[float, Dict[str, Any]]],
    max_per_article: int = 3,
) -> List[Tuple[float, Dict[str, Any]]]:
    article_count: Dict[str, int] = defaultdict(int)
    filtered: List[Tuple[float, Dict[str, Any]]] = []

    for score, chunk in chunks:
        art = chunk.get("parent_article_no") or chunk.get("article_no")
        if article_count[art] < max_per_article:
            filtered.append((score, chunk))
            article_count[art] += 1

    return filtered


def fetch_parents(
    chunks: List[Tuple[float, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    parent_ids = set()
    for _, chunk in chunks:
        depth = _infer_depth(chunk)
        cid = chunk.get("id", "")
        if depth >= 1:
            parent_ids.add(_path_up(cid, 1))
        if depth >= 2:
            parent_ids.add(_path_up(cid, 2))
    parent_ids.discard(None)
    if not parent_ids:
        return {}
    return fetch_chunks_by_ids(list(parent_ids))


def fetch_siblings(
    chunks: List[Tuple[float, Dict[str, Any]]],
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    siblings_by_parent: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = defaultdict(list)
    for score, chunk in chunks:
        parent_art = chunk.get("parent_article_no") or chunk.get("article_no")
        chunk_type = chunk.get("chunk_type")
        doc_name = chunk.get("doc_name")
        if not parent_art or not chunk_type:
            continue
        siblings_by_parent[(doc_name, parent_art, chunk_type)].append(
            (score, chunk.get("id", ""), chunk)
        )

    result: Dict[str, List[Dict[str, Any]]] = {}
    for (doc_name, parent_art, chunk_type), siblings in siblings_by_parent.items():
        sorted_siblings = sorted(siblings, key=lambda x: x[0], reverse=True)
        top_siblings = sorted_siblings[:top_n]
        for _, sibling_id, sib_chunk in top_siblings:
            parent_path = _path_up(sibling_id, 1) if chunk_type != "article" else None
            if parent_path:
                result.setdefault(parent_path, []).append({
                    "article_no": sib_chunk.get("article_no") or sib_chunk.get("parent_article_no"),
                    "article_title": sib_chunk.get("article_title"),
                    "chunk_type": chunk_type,
                })
    return result


def _count_tokens(text: str) -> int:
    return len(text) // 4


def assemble_context(
    query_type: str,
    reranked_chunks: List[Tuple[float, Dict[str, Any]]],
    parent_chunks: Dict[str, Dict[str, Any]],
    sibling_info: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    article_filter: Optional[str] = None,
) -> str:
    token_limits = {
        "definitional": 1500,
        "provision": 4000,
        "conceptual": 2500,
    }

    priorities = {
        "definitional": ["chunks", "parents"],
        "provision": ["parents", "chunks", "siblings"] if article_filter else ["chunks", "parents", "siblings"],
        "conceptual": ["parents", "chunks"] if article_filter else ["chunks", "parents"],
    }

    limit = token_limits.get(query_type, 2500)
    priority_order = priorities.get(query_type, ["chunks", "parents"])

    context_parts: List[str] = []
    used_tokens = 0
    remaining = lambda: limit - used_tokens

    def _add(text: str) -> bool:
        nonlocal used_tokens
        tokens = _count_tokens(text)
        if tokens > remaining():
            return False
        context_parts.append(text)
        used_tokens += tokens
        return True

    if "chunks" in priority_order:
        for _, chunk in reranked_chunks:
            cid = chunk.get("id", "")
            text = chunk.get("text", "")
            if not text:
                continue
            entry = f"[{cid}]: {text}"
            if not _add(entry):
                break

    if "parents" in priority_order:
        for parent_id in sorted(parent_chunks.keys()):
            parent = parent_chunks[parent_id]
            text = parent.get("text", "")
            if not text:
                continue
            entry = f"[{parent_id}]: {text}"
            if not _add(entry):
                break

    if sibling_info and "siblings" in priority_order:
        for parent_id, siblings in sibling_info.items():
            for i, sib in enumerate(siblings):
                if i < 3:
                    art_no = sib.get("article_no", "")
                    art_title = sib.get("article_title", "")
                    entry = f"[{parent_id}/sibling]: Article {art_no}"
                    if art_title:
                        entry += f" \u2013 {art_title}"
                    if not _add(entry):
                        break
                else:
                    art_no = sib.get("article_no", "")
                    entry = f"[Article {art_no}]"
                    if not _add(entry):
                        break

    return "\n\n".join(context_parts)


def compute_retrieval_confidence(
    reranked_chunks: List[Tuple[float, Dict[str, Any]]],
) -> float:
    if not reranked_chunks:
        return 0.0
    scores = [s for s, _ in reranked_chunks]
    return round(float(np.mean(scores)), 4)
