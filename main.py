"""FastAPI application."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app import config, db
from app.graph import get_graph
from app.ingestion import IngestionError, index_documents, ingest_corpus_dir, load_text, load_url
from app.schemas import DocumentInfo, FeedbackRequest, QueryRequest, QueryResponse
from app.vectorstore import count_chunks, list_documents

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

UNGROUNDED_WARNING = "\n\n⚠️ Note: this answer could not be fully verified against the documentation."


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    try:
        if count_chunks() == 0:
            log.info("Vector store empty - ingesting bundled corpus")
            ingest_corpus_dir()
    except IngestionError as e:
        log.warning("Startup ingestion skipped: %s", e)
    yield


app = FastAPI(title="RAG Technical Documentation Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": count_chunks(), "llm_configured": bool(config.GROQ_API_KEY)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not config.GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not configured on the server.")
    if count_chunks() == 0:
        raise HTTPException(409, "No documents indexed yet. POST /ingest first.")
    try:
        state = get_graph().invoke({"question": req.question})
    except Exception as e:
        log.exception("graph failure")
        raise HTTPException(502, f"Upstream LLM/workflow error: {type(e).__name__}") from e

    answer = state["generation"]
    if state.get("grounded") is False:
        answer += UNGROUNDED_WARNING
    qid = db.log_query(req.question, answer)
    return QueryResponse(
        query_id=qid,
        answer=answer,
        sources=state.get("sources", []),
        query_type=state.get("query_type"),
        rewritten_query=state.get("query"),
        retries=state.get("retries", 0),
        grounded=state.get("grounded"),
    )


@app.post("/ingest", status_code=201)
async def ingest(files: list[UploadFile] = File(default=[]), urls: str = Form(default="")):
    """Upload .md/.txt/.html files and/or pass URLs (comma or newline separated)."""
    url_list = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]
    if not files and not url_list:
        raise HTTPException(400, "Provide at least one file or URL.")
    docs, errors = [], []
    for f in files:
        try:
            docs.append(load_text(f.filename or "upload.txt", (await f.read()).decode("utf-8", errors="ignore")))
        except IngestionError as e:
            errors.append(str(e))
    for u in url_list:
        try:
            docs.append(load_url(u))
        except IngestionError as e:
            errors.append(str(e))
    if not docs:
        raise HTTPException(422, {"message": "No documents could be ingested.", "errors": errors})
    return {"ingested": index_documents(docs), "errors": errors}


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@app.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest):
    if not db.query_exists(req.query_id):
        raise HTTPException(404, "Unknown query_id.")
    db.save_feedback(req.query_id, req.rating, req.comment)
    return {"status": "recorded"}
