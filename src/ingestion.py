import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chromadb
from chromadb.config import Settings as ChromaSettings

from settings import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, EMBEDDING_MAX_LENGTH, EMBEDDING_BATCH_SIZE
from src.chunker import chunk_document, NonEnglishDocumentError

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        if EMBEDDING_MAX_LENGTH > 0:
            _model.max_seq_length = EMBEDDING_MAX_LENGTH
        print(f"  Model: {EMBEDDING_MODEL} | Dim: {_model.get_sentence_embedding_dimension()} | Max seq: {_model.max_seq_length}")
    return _model


def encode_texts(texts, show_progress=True):
    model = _get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=show_progress,
    )


def flatten_metadata(chunk_dict):
    metadata = {}
    for key, value in chunk_dict.items():
        if key == "text":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            metadata[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            metadata[key] = value
    return metadata


def _get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection(client):
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def _load_existing_ids(collection):
    try:
        return set(collection.get(include=[])["ids"])
    except Exception:
        return set()


def _should_skip(chunk):
    if chunk.get("is_structural"):
        return True
    if chunk.get("text") is None:
        return True
    return False


def store_chunks(chunks_with_embeddings, collection, existing_ids, batch_size=64):
    inserted = 0
    skipped_dup = 0
    skipped_structural = 0
    skipped_notext = 0

    ids, embeddings, documents, metadatas = [], [], [], []

    def _flush():
        nonlocal inserted
        if not ids:
            return
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        inserted += len(ids)
        ids.clear()
        embeddings.clear()
        documents.clear()
        metadatas.clear()

    for chunk in chunks_with_embeddings:
        if _should_skip(chunk):
            if chunk.get("is_structural"):
                skipped_structural += 1
            else:
                skipped_notext += 1
            continue
        path = chunk.get("path")
        if path in existing_ids:
            skipped_dup += 1
            continue
        ids.append(path)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(flatten_metadata(chunk))
        if len(ids) >= batch_size:
            _flush()
    _flush()
    return inserted, skipped_dup, skipped_structural, skipped_notext


def ingest_document(filepath, collection, existing_ids):
    chunks = chunk_document(filepath)

    embeddable = [c for c in chunks if not _should_skip(c)]
    skipped_structural = len(chunks) - len(embeddable)

    seen_in_doc = set()
    to_embed = []
    skipped_dup = 0
    for c in embeddable:
        p = c["path"]
        if p in existing_ids or p in seen_in_doc:
            skipped_dup += 1
            continue
        seen_in_doc.add(p)
        to_embed.append(c)

    if to_embed:
        texts = [c["text"] for c in to_embed]
        vectors = encode_texts(texts)
        for c, vec in zip(to_embed, vectors):
            c["embedding"] = vec.tolist()

    inserted, dup2, struct2, notext2 = store_chunks(to_embed, collection, existing_ids)

    existing_ids.update(c["path"] for c in to_embed)

    return {
        "inserted": inserted,
        "skipped_dup": skipped_dup + dup2,
        "skipped_structural": skipped_structural + struct2,
        "skipped_notext": notext2,
        "total": len(chunks),
    }
