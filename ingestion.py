"""Document ingestion: load -> clean -> chunk -> embed -> store.

Chunking strategy (see README for rationale):
  1. Markdown docs are first split on headings (#, ##, ###) so a chunk never
     straddles two unrelated sections and every chunk knows its section path.
  2. Each section is then split with a Markdown-aware recursive splitter
     (900 chars, 150 overlap) that prefers breaking at headings / code fences /
     paragraphs before sentences.
  3. The section path is prepended to each chunk's text so the embedding
     carries context such as "Request Body > Declare your data model".
"""
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app import config

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class IngestionError(ValueError):
    pass


# ---------- loading ----------
def clean_markdown(text: str) -> str:
    """Strip mkdocs-only noise that would pollute embeddings."""
    text = re.sub(r"^\{\*.*?\*\}\s*$", "", text, flags=re.M)      # {* ../../docs_src/x.py *} includes
    text = re.sub(r"^///.*$", "", text, flags=re.M)               # /// note admonition fences
    text = re.sub(r"\s*\{ #[\w\-]+ \}", "", text)                 # heading anchors "{ #id }"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n")).strip()


def load_text(name: str, content: str, doc_type: str | None = None) -> Document:
    ext = doc_type or Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")
    if ext in {".html", ".htm"}:
        content, ext = html_to_text(content), ".txt"
    elif ext == ".md":
        content = clean_markdown(content)
    if not content.strip():
        raise IngestionError(f"'{name}' is empty after cleaning.")
    return Document(page_content=content, metadata={"source": name, "doc_type": ext})


def load_file(path: Path) -> Document:
    return load_text(path.name, path.read_text(encoding="utf-8", errors="ignore"))


def load_url(url: str) -> Document:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise IngestionError(f"Invalid URL: {url}")
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise IngestionError(f"Could not fetch {url}: {e}") from e
    path = parsed.path.strip("/") or parsed.netloc
    name = "/".join(path.split("/")[-2:])
    ext = Path(parsed.path).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:  # most doc pages are HTML without an extension
        ext = ".html" if "html" in resp.headers.get("content-type", "") else ".txt"
    return load_text(name, resp.text, ext)


# ---------- chunking ----------
def chunk_document(doc: Document) -> list[Document]:
    if doc.metadata.get("doc_type") == ".md":
        sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(doc.page_content)
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
    else:
        sections = [Document(page_content=doc.page_content, metadata={})]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )

    chunks: list[Document] = []
    for piece in splitter.split_documents(sections):
        path = " > ".join(piece.metadata[k] for k in ("h1", "h2", "h3") if k in piece.metadata)
        text = f"[Section: {path}]\n{piece.page_content}" if path else piece.page_content
        chunks.append(Document(page_content=text, metadata={**doc.metadata, "section": path}))

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["chunk_id"] = f"{c.metadata['source']}::{i}"
    return chunks


# ---------- indexing ----------
def index_documents(docs: list[Document]) -> list[dict]:
    """Chunk + embed + store. Re-ingesting a source replaces its old chunks (idempotent)."""
    from app.vectorstore import get_vectorstore

    vs = get_vectorstore()
    report = []
    for doc in docs:
        chunks = chunk_document(doc)
        vs._collection.delete(where={"source": doc.metadata["source"]})
        vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        report.append({"source": doc.metadata["source"], "chunks": len(chunks)})
    return report


def ingest_corpus_dir(directory: Path = config.CORPUS_DIR) -> list[dict]:
    files = sorted(p for p in directory.glob("*") if p.suffix.lower() in config.ALLOWED_EXTENSIONS)
    if not files:
        raise IngestionError(f"No documents found in {directory}. Run scripts/fetch_corpus.py first.")
    return index_documents([load_file(p) for p in files])
