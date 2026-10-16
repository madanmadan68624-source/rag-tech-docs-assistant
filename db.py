"""Tiny SQLite store for query logs + feedback."""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from app import config


@contextmanager
def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY, question TEXT, answer TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, rating TEXT,
                comment TEXT, created_at TEXT);
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_query(question: str, answer: str) -> str:
    qid = uuid.uuid4().hex
    with _conn() as c:
        c.execute("INSERT INTO queries VALUES (?,?,?,?)", (qid, question, answer, _now()))
    return qid


def query_exists(query_id: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM queries WHERE id=?", (query_id,)).fetchone() is not None


def save_feedback(query_id: str, rating: str, comment: str | None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO feedback (query_id, rating, comment, created_at) VALUES (?,?,?,?)",
            (query_id, rating, comment, _now()),
        )
