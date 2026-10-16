"""Central configuration (env-driven, sensible defaults)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
DB_PATH = BASE_DIR / "data" / "app.db"

COLLECTION_NAME = "tech_docs"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))          # query-rewrite retries
MAX_GEN_ATTEMPTS = int(os.getenv("MAX_GEN_ATTEMPTS", "2"))  # generation attempts (hallucination loop)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}
