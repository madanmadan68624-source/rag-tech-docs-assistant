"""LLM-free tests: chunking + graph wiring/routing (nodes are monkeypatched)."""
from langchain_core.documents import Document

from app import config, graph, ingestion


def test_markdown_chunking_keeps_section_context():
    md = "# Title\n\nintro\n\n## Body\n\n" + ("word " * 400) + "\n\n## Other\n\nshort"
    doc = ingestion.load_text("t.md", md)
    chunks = ingestion.chunk_document(doc)
    assert len(chunks) >= 3
    assert all(c.metadata["source"] == "t.md" for c in chunks)
    assert any("[Section: Title > Body]" in c.page_content for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_clean_markdown_strips_noise():
    out = ingestion.clean_markdown("# H { #h }\n\n{* ../x.py *}\n\n/// note\ntext\n///\n")
    assert "{*" not in out and "///" not in out and "{ #" not in out


def test_unsupported_type_rejected():
    import pytest
    with pytest.raises(ingestion.IngestionError):
        ingestion.load_text("a.exe", "x")


def _patch(monkeypatch, relevant_on_call):
    calls = {"retrieve": 0, "rewrite": 0}

    monkeypatch.setattr(graph, "analyze_query", lambda s: {"query": "q", "query_type": "how-to", "retries": 0, "gen_attempts": 0, "grounded": None})

    def retrieve(s):
        calls["retrieve"] += 1
        return {"documents": [Document(page_content="c", metadata={"source": "a.md"})]}
    monkeypatch.setattr(graph, "retrieve", retrieve)

    monkeypatch.setattr(graph, "grade_documents", lambda s: {"found_answer": calls["retrieve"] >= relevant_on_call,
                                                              "documents": s["documents"]})

    def rewrite(s):
        calls["rewrite"] += 1
        return {"query": "q2", "retries": s["retries"] + 1}
    monkeypatch.setattr(graph, "rewrite_query", rewrite)
    monkeypatch.setattr(graph, "generate", lambda s: {"generation": "answer [1]", "sources": [], "gen_attempts": s["gen_attempts"] + 1})
    monkeypatch.setattr(graph, "check_hallucination", lambda s: {"grounded": True})
    monkeypatch.setattr(graph, "no_answer", lambda s: {"generation": graph.NO_ANSWER_TEXT, "sources": []})
    return calls


def test_retry_then_success(monkeypatch):
    calls = _patch(monkeypatch, relevant_on_call=2)
    out = graph.build_graph().invoke({"question": "q"})
    assert out["generation"] == "answer [1]"
    assert calls == {"retrieve": 2, "rewrite": 1}


def test_gives_up_after_max_retries(monkeypatch):
    calls = _patch(monkeypatch, relevant_on_call=99)
    out = graph.build_graph().invoke({"question": "q"})
    assert out["generation"] == graph.NO_ANSWER_TEXT
    assert calls["rewrite"] == config.MAX_RETRIES
    assert calls["retrieve"] == config.MAX_RETRIES + 1


def test_hallucination_loop_regenerates_once(monkeypatch):
    _patch(monkeypatch, relevant_on_call=1)
    seen = []
    monkeypatch.setattr(graph, "check_hallucination", lambda s: seen.append(1) or {"grounded": False})
    out = graph.build_graph().invoke({"question": "q"})
    assert out["gen_attempts"] == config.MAX_GEN_ATTEMPTS and len(seen) == config.MAX_GEN_ATTEMPTS
