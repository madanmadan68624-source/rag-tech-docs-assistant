# RAG Technical Documentation Assistant

A Retrieval-Augmented Generation system with a **self-corrective LangGraph workflow**, served via **FastAPI**.
It answers questions about technical documentation (demo corpus: 5 pages of the FastAPI tutorial) with inline citations, and says *"I don't know"* instead of guessing when the docs don't contain the answer.

**Stack:** Python · LangGraph · FastAPI · ChromaDB · sentence-transformers (local embeddings) · Groq (`llama-3.3-70b-versatile`, free tier)

## Architecture

```mermaid
flowchart TD
    S([START]) --> A[analyze_query<br/>rewrite + classify]
    A --> R[retrieve<br/>Chroma top-k]
    R --> G[grade_documents<br/>LLM per-chunk yes/no]
    G -->|relevant chunks found| GEN[generate<br/>answer + citations]
    G -->|none relevant & retries left| W[rewrite_query]
    W --> R
    G -->|none relevant & retries exhausted| N[no_answer<br/>"I don't know"]
    GEN --> H[check_hallucination<br/>bonus: Self-RAG style]
    H -->|grounded| E([END])
    H -->|not grounded & attempts left| GEN
    H -->|still not grounded| E
    N --> E
```

### State schema (`app/graph.py::GraphState`)
| Field | Purpose |
|---|---|
| `question` | Original question, **never mutated** (graders/generator always judge against what the user asked) |
| `query` | Current retrieval query; overwritten by `analyze_query` and each `rewrite_query` |
| `query_type` | conceptual / how-to / troubleshooting / api-reference – steers answer style |
| `documents` | Retrieved chunks, replaced by the *filtered* relevant set after grading |
| `retries` | Number of query rewrites so far → bounded by `MAX_RETRIES` (default 2) |
| `gen_attempts` | Generation attempts → bounds the hallucination-regeneration loop (`MAX_GEN_ATTEMPTS`, default 2) |
| `found_answer` | Grading outcome used by the conditional edge |
| `generation`, `sources`, `grounded` | Final answer, citation metadata, hallucination-check verdict |

Retry termination is guaranteed: every cycle (`rewrite_query → retrieve`, `check_hallucination → generate`) increments a counter that the routing function checks.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your free GROQ_API_KEY (https://console.groq.com)
python -m scripts.fetch_corpus                         # optional: corpus is already in data/corpus
python -m scripts.ingest                               # optional: auto-runs on first server start if index is empty
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for the interactive Swagger UI. Run tests (no API key needed): `pytest -q`.

## API & examples

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/query` | Ask a question → answer + sources |
| POST | `/ingest` | Upload `.md/.txt/.html` files and/or pass URLs |
| GET | `/documents` | List indexed documents with chunk counts |
| POST | `/feedback` | 👍/👎 + optional comment for a `query_id` |
| GET | `/health` | Liveness + index size |

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How do I declare a request body in FastAPI?"}'
```
```json
{
  "query_id": "3f9c...",
  "answer": "Declare a Pydantic model that inherits from `BaseModel` and use it as a parameter type in your path operation function [1][2]. ...",
  "sources": [{"id": 1, "source": "fastapi_body.md", "section": "Request Body > Create your data model",
               "chunk_id": "fastapi_body.md::3", "snippet": "...", "cited": true}],
  "query_type": "how-to",
  "rewritten_query": "declare request body Pydantic BaseModel FastAPI POST parameter",
  "retries": 0,
  "grounded": true
}
```
Out-of-scope question (`"What is the capital of France?"`) → graph rewrites twice, finds nothing relevant, returns *"I don't know based on the indexed documentation..."* with `sources: []`.

```bash
curl -X POST localhost:8000/ingest -F "files=@my_notes.md" -F "urls=https://docs.pydantic.dev/latest/concepts/models/"
curl localhost:8000/documents
curl -X POST localhost:8000/feedback -H "Content-Type: application/json" \
  -d '{"query_id": "3f9c...", "rating": "up", "comment": "clear answer"}'
```
Status codes: `201` created, `400` nothing to ingest, `404` unknown `query_id`, `409` empty index, `422` validation/unsupported file, `502` LLM failure, `503` missing API key.

## Design decisions & tradeoffs

**Chunking** (`app/ingestion.py`): two-stage. (1) Split Markdown on `#/##/###` headings so chunks never mix unrelated sections; (2) split sections with a Markdown-aware recursive splitter, **900 chars / 150 overlap**, preferring heading → code-fence → paragraph boundaries. The section path (`[Section: Request Body > Create your data model]`) is prepended to each chunk so embeddings carry context. *Why:* technical docs are heading-structured and code blocks must not be sliced mid-example; ~900 chars ≈ one concept + its snippet. *Tradeoff:* tiny sections become tiny chunks; very long code blocks can still be split.

**Cleaning:** mkdocs-only syntax (`{* file.py *}` includes, `///` admonitions, `{ #anchor }`) is stripped because it adds noise to embeddings.

**Embeddings:** `all-MiniLM-L6-v2` locally – free, no API key, fast on CPU, cosine similarity. *Tradeoff:* weaker than OpenAI/BGE-large on nuanced queries; swap via `EMBEDDING_MODEL`.

**Why grade with an LLM, per chunk:** vector similarity can't tell "mentions the same words" from "answers the question". Grading runs in parallel (`batch`, concurrency 4) against the *original* question. If the grader itself errors, the chunk is kept (fail-open) so an LLM glitch doesn't produce false "I don't know"s.

**Query analysis:** one structured-output call yields both the expanded query and a query type; falls back to the raw question on failure.

**Citations:** context chunks are numbered `[1]..[n]`; the model cites inline; `sources[].cited` flags which were actually used. Original numbering is preserved so citations always map correctly.

**Hallucination check (bonus):** a Self-RAG-style node verifies the answer is supported by the context; one stricter regeneration is allowed, after which the API appends a warning and returns `grounded: false` rather than silently shipping an unverified answer.

**Storage:** Chroma (persistent, local) for vectors; SQLite for query log + feedback (so feedback can be validated against real `query_id`s and survive restarts). Re-ingesting a source replaces its chunks (idempotent).

**Sync endpoints:** the graph is blocking, so `/query` is a plain `def` and FastAPI runs it in a threadpool.

## Assumptions
- Corpus is English Markdown/HTML/text; single-tenant, local use; no auth.
- Groq's free-tier rate limits are adequate for demo traffic (grading makes up to `TOP_K` parallel calls per retrieval).
- Chunk-level relevance (not full-document) is the right grading granularity.

## What I'd improve with more time
- Hybrid retrieval (BM25 + vectors) and a cross-encoder reranker; metadata filters per source.
- Conversation memory (LangGraph checkpointer keyed by `session_id`) + follow-up question condensation.
- Web-search fallback (Tavily) before `no_answer`.
- Evaluation harness (golden Q&A set; measure retrieval hit-rate, faithfulness) and CI.
- Async/streaming responses, auth + rate limiting, background ingestion for large files, Docker image.
- Smarter handling of oversized code blocks in the chunker.

## Project layout
```
app/        config.py · ingestion.py · vectorstore.py · graph.py · main.py · schemas.py · db.py
scripts/    fetch_corpus.py · ingest.py
data/corpus/  5 FastAPI tutorial pages (Markdown)
tests/      chunking + graph routing tests (LLM-free)
```
