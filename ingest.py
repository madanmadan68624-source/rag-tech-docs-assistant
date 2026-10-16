"""Standalone ingestion: python -m scripts.ingest"""
from app.ingestion import ingest_corpus_dir

if __name__ == "__main__":
    for row in ingest_corpus_dir():
        print(f"indexed {row['source']}: {row['chunks']} chunks")
