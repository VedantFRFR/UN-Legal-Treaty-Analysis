import os
import json
import sys
import argparse

from settings import DATA_DIR, OUTPUT_DIR
from src.chunker import chunk_document, NonEnglishDocumentError


def _resolve_files(args):
    if args.file:
        files = [f if os.path.isabs(f) else os.path.join(DATA_DIR, f) for f in args.file]
    else:
        if not os.path.isdir(DATA_DIR):
            print(f"Data directory not found: {DATA_DIR}")
            sys.exit(1)
        files = sorted(
            os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")
        )
    return [f for f in files if os.path.isfile(f)]


def _chunk_and_report(filepath, output_dir=None):
    doc_name = os.path.basename(filepath)
    try:
        chunks = chunk_document(filepath)
    except (NonEnglishDocumentError, Exception) as e:
        prefix = "SKIPPED (non-English)" if isinstance(e, NonEnglishDocumentError) else "ERROR"
        print(f"  {prefix}: {e}")
        return None
    print(f"  -> {len(chunks)} chunks")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, doc_name.replace(".pdf", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved: {out_path}")
    return chunks


def _run_chunk_mode(files, output_dir):
    results = {}
    for fp in files:
        print(f"\n--- {os.path.basename(fp)} ---")
        chunks = _chunk_and_report(fp, output_dir)
        if chunks is not None:
            results[os.path.basename(fp)] = chunks
    print(f"\n{'='*40}\nChunked {len(results)} / {len(files)} documents.")


def _run_ingest_mode(files):
    from src.ingestion import ingest_document, _get_client, _get_collection, _load_existing_ids

    client = _get_client()
    collection = _get_collection(client)
    existing_ids = _load_existing_ids(collection)

    totals = {"inserted": 0, "skipped_dup": 0, "skipped_structural": 0, "skipped_notext": 0}
    for fp in files:
        doc_name = os.path.basename(fp)
        print(f"\n--- {doc_name} ---")
        try:
            result = ingest_document(fp, collection, existing_ids)
        except NonEnglishDocumentError as e:
            print(f"  SKIPPED (non-English): {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        for k in totals:
            totals[k] += result[k]
        print(f"  Inserted: {result['inserted']} | Dups: {result['skipped_dup']} "
              f"| Structural: {result['skipped_structural']} | No-text: {result['skipped_notext']}")
    print(f"\n{'='*40}")
    print(f"Inserted: {totals['inserted']} | Dups: {totals['skipped_dup']} "
          f"| Structural: {totals['skipped_structural']} | No-text: {totals['skipped_notext']}")


def _run_query_mode(args):
    from src.ingestion import _get_client, _get_collection

    try:
        collection = _get_collection(_get_client())
    except Exception:
        print("No collection found. Run ingestion first.")
        return

    count = collection.count()
    print(f"Collection: treaty_chunks | Total stored: {count}")
    if count == 0:
        return

    meta = collection.get(include=["metadatas"])
    docs, types = {}, {}
    for m in meta["metadatas"]:
        docs[m.get("doc_name", "?")] = docs.get(m.get("doc_name", "?"), 0) + 1
        types[m.get("chunk_type", "?")] = types.get(m.get("chunk_type", "?"), 0) + 1

    print("By document:", *(f"  {k}: {v}" for k, v in sorted(docs.items())), sep="\n")
    print("By type:", *(f"  {k}: {v}" for k, v in sorted(types.items())), sep="\n")

    if args.file:
        results = collection.get(
            where={"doc_name": os.path.basename(args.file[0])},
            limit=5, include=["documents", "metadatas"],
        )
        if results["ids"]:
            print(f"\nSample chunks from {args.file[0]}:")
            for i, pid in enumerate(results["ids"]):
                text = results["documents"][i][:100]
                print(f"  [{results['metadatas'][i].get('chunk_type','?')}] {pid}: {text}...")


def _check_retrieval_prerequisites():
    from dotenv import load_dotenv
    load_dotenv()

    try:
        from src.ingestion import _get_client, _get_collection
        count = _get_collection(_get_client()).count()
        if count == 0:
            print("ERROR: ChromaDB is empty. Run 'python main.py --mode ingest' first.")
            return False
        print(f"  ChromaDB: {count} chunks")
    except Exception:
        print("ERROR: ChromaDB collection not found. Run 'python main.py --mode ingest' first.")
        return False

    key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not found in .env or environment.")
        return False
    print(f"  Groq API key: {key[:8]}...")
    return True


def _rebuild_bm25():
    from src.retrieval import rebuild_bm25
    try:
        rebuild_bm25()
    except Exception as e:
        print(f"  BM25 rebuild skipped: {e}")


def _run_retrieval_mode(query, rebuild_bm25=False):
    from src.agent import run_pipeline

    if rebuild_bm25:
        _rebuild_bm25()

    print(f"\nQuery: {query}\n{'='*60}")
    try:
        result = run_pipeline(query)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nType: {result['query_type']} | Confidence: {result['retrieval_confidence']} "
              f"| Chunks: {result['chunks_retrieved']} -> {result['chunks_post_filter']}")
    except Exception as e:
        print(f"Error: {e}")


def _run_serve_mode():
    import os as _os
    import json
    import shutil
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, UploadFile, File, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    UPLOADS_DIR = _os.path.join(DATA_DIR, "uploads")
    _os.makedirs(UPLOADS_DIR, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app):
        try:
            from src.retrieval import _build_bm25_index
            _build_bm25_index()
            print("BM25 index loaded.")
        except Exception as e:
            print(f"BM25 index not built (run ingestion first): {e}")
        yield

    app = FastAPI(title="Legal Analysis", lifespan=lifespan)
    templates_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "templates")
    templates = Jinja2Templates(directory=templates_dir)

    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        answer: str
        query_type: str
        retrieval_confidence: float
        chunks_retrieved: int
        chunks_post_filter: int
        error: str = ""

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.post("/query", response_model=QueryResponse)
    async def handle_query(req: QueryRequest):
        if not req.query.strip():
            return QueryResponse(answer="", query_type="", retrieval_confidence=0.0,
                                 chunks_retrieved=0, chunks_post_filter=0, error="Empty query")
        try:
            from src.agent import run_pipeline
            result = run_pipeline(req.query)
            return QueryResponse(
                answer=result.get("answer", ""),
                query_type=result.get("query_type", "conceptual"),
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                chunks_retrieved=result.get("chunks_retrieved", 0),
                chunks_post_filter=result.get("chunks_post_filter", 0),
                error=result.get("error") or "",
            )
        except Exception as e:
            return QueryResponse(answer="", query_type="", retrieval_confidence=0.0,
                                 chunks_retrieved=0, chunks_post_filter=0, error=str(e))

    @app.post("/upload")
    async def upload_documents(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(400, "No files provided")

        results = []
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                results.append({
                    "doc_name": file.filename or "unknown",
                    "status": "error",
                    "error": "Only PDF files are allowed",
                })
                continue

            dest = _os.path.join(UPLOADS_DIR, file.filename)
            with open(dest, "wb") as f:
                content = await file.read()
                f.write(content)

            try:
                from src.ingestion import _get_client, _get_collection, _load_existing_ids, ingest_document
                client = _get_client()
                collection = _get_collection(client)
                existing_ids = _load_existing_ids(collection)
                result = ingest_document(dest, collection, existing_ids)
                results.append({
                    "doc_name": file.filename,
                    "status": "success",
                    "chunks": result["total"],
                    "inserted": result["inserted"],
                    "skipped_dup": result["skipped_dup"],
                })
            except Exception as e:
                if _os.path.exists(dest):
                    _os.remove(dest)
                results.append({
                    "doc_name": file.filename,
                    "status": "error",
                    "error": str(e),
                })

        if any(r["status"] == "success" for r in results):
            _rebuild_bm25()

        return {"results": results}

    @app.get("/documents")
    async def list_documents():
        from src.ingestion import _get_client, _get_collection
        try:
            collection = _get_collection(_get_client())
            meta = collection.get(include=["metadatas"])
        except Exception:
            return {"documents": [], "total_chunks": 0}
        docs = {}
        for m in meta["metadatas"]:
            name = m.get("doc_name", "?")
            docs[name] = docs.get(name, 0) + 1
        return {"documents": [{"name": k, "chunks": v} for k, v in sorted(docs.items())],
                "total_chunks": len(meta["metadatas"])}

    @app.delete("/documents/{name:path}")
    async def delete_document(name: str):
        from src.ingestion import _get_client, _get_collection
        try:
            collection = _get_collection(_get_client())
        except Exception:
            raise HTTPException(404, "No collection found")
        count = collection.delete(where={"doc_name": name})
        filepath = _os.path.join(UPLOADS_DIR, name)
        if _os.path.exists(filepath):
            _os.remove(filepath)
        _rebuild_bm25()
        return {"status": "deleted", "doc_name": name, "chunks_removed": count}

    @app.post("/corpus/clear")
    async def clear_corpus():
        from src.ingestion import _get_client, _get_collection
        try:
            collection = _get_collection(_get_client())
            count = collection.count()
            all_ids = collection.get(include=[])["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
        except Exception:
            count = 0

        if _os.path.isdir(UPLOADS_DIR):
            for fname in _os.listdir(UPLOADS_DIR):
                fpath = _os.path.join(UPLOADS_DIR, fname)
                try:
                    if _os.path.isfile(fpath):
                        _os.remove(fpath)
                except Exception:
                    pass

        _rebuild_bm25()

        return {"status": "cleared", "chunks_removed": count}

    port = int(_os.environ.get("LEGAL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description="Legal Analysis pipeline")
    parser.add_argument("--file", "-f", nargs="+", help="PDF file(s) to process (default: all in data/)")
    parser.add_argument("--mode", choices=["chunk", "ingest", "query", "retrieval", "pipeline", "serve"],
                        default="chunk", help="Stage to run (default: chunk)")
    parser.add_argument("--save", "-s", action="store_true", help="Save chunk JSON to output/ (chunk mode)")
    parser.add_argument("--query", "-q", type=str, help="Query for retrieval or pipeline mode")
    parser.add_argument("--rebuild-bm25", action="store_true", help="Rebuild BM25 index before retrieval")
    args = parser.parse_args()

    if args.mode == "query":
        _run_query_mode(args)
        return

    if args.mode == "serve":
        _run_serve_mode()
        return

    if args.mode == "retrieval":
        if not _check_retrieval_prerequisites():
            sys.exit(1)
        _run_retrieval_mode(args.query, rebuild_bm25=args.rebuild_bm25)
        return

    if args.mode == "pipeline":
        files = _resolve_files(args)
        if not files:
            print("No PDF files found.")
            sys.exit(1)
        print(f"\nPipeline: chunk ({len(files)} PDFs) -> ingest -> BM25 -> retrieval")
        _run_chunk_mode(files, output_dir=OUTPUT_DIR)
        _run_ingest_mode(files)
        _rebuild_bm25()
        if args.query:
            if not _check_retrieval_prerequisites():
                sys.exit(1)
            _run_retrieval_mode(args.query)
        else:
            print("\nPipeline complete. Run 'python main.py --mode retrieval' for Q&A.")
        return

    files = _resolve_files(args)
    if not files:
        print("No PDF files found.")
        sys.exit(1)

    if args.mode == "chunk":
        _run_chunk_mode(files, OUTPUT_DIR if args.save else None)
    else:
        _run_ingest_mode(files)


if __name__ == "__main__":
    main()
