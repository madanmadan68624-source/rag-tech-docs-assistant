"""Vector store access. Heavy imports are lazy so unit tests stay light."""
from collections import Counter
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_chunks() -> int:
    return get_vectorstore()._collection.count()


def list_documents() -> list[dict]:
    """Aggregate chunk metadata into one row per source document."""
    data = get_vectorstore().get(include=["metadatas"])
    counts = Counter(m.get("source", "unknown") for m in data["metadatas"])
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]
