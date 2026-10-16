"""Self-corrective RAG workflow built with LangGraph.

    START -> analyze_query -> retrieve -> grade_documents --(relevant)--> generate -> check_hallucination --(grounded)--> END
                                  ^               |                          ^                |
                                  |        (none relevant,                   +--(not grounded, |
                                  |         retries left)                       attempts left) +
                                  +---- rewrite_query <--+
                                                |
                                  (none relevant, no retries left) -> no_answer -> END
"""
import logging
import re
from functools import lru_cache
from typing import List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app import config

log = logging.getLogger(__name__)

NO_ANSWER_TEXT = (
    "I don't know based on the indexed documentation. I couldn't find relevant "
    "information for this question, even after rewriting the query."
)


# ---------------------------------------------------------------- state
class GraphState(TypedDict, total=False):
    question: str                 # original user question (never mutated)
    query: str                    # current retrieval query (rewritten by analyze/rewrite nodes)
    query_type: str               # conceptual | how-to | troubleshooting | api-reference
    documents: List[Document]     # retrieved chunks; replaced by the filtered set after grading
    retries: int                  # number of query rewrites performed so far
    gen_attempts: int             # number of generation attempts (for hallucination loop)
    generation: str
    sources: List[dict]
    grounded: Optional[bool]      # result of hallucination check
    found_answer: bool


# ---------------------------------------------------------------- structured outputs
class QueryAnalysis(BaseModel):
    rewritten_query: str = Field(description="Search query expanded with synonyms/keywords, self-contained.")
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api-reference"]


class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' if the chunk helps answer the question")


class GradeGrounded(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' if every claim is supported by the context")


# ---------------------------------------------------------------- prompts
ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You prepare user questions for semantic search over technical documentation. "
     "Rewrite the question into a clear, self-contained search query: fix ambiguity, add relevant "
     "technical synonyms and key terms, keep it under 30 words. Also classify it as one of: "
     "conceptual, how-to, troubleshooting, api-reference."),
    ("human", "Question: {question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a retrieved documentation chunk is relevant to a user question. "
     "Answer 'yes' if the chunk contains keywords or meaning that help answer the question, "
     "even partially. Answer 'no' if it is off-topic. Be lenient on wording, strict on topic."),
    ("human", "Question: {question}\n\nChunk:\n{document}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The previous search query returned no relevant documentation. Write a DIFFERENT query for the "
     "same information need: use alternative terminology, broader or more specific concepts. "
     "Return only the new query text."),
    ("human", "Original question: {question}\nPrevious query: {query}\nNew query:"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical documentation assistant. Answer the question using ONLY the numbered context "
     "below. Rules:\n"
     "- Cite sources inline like [1] or [2][3] right after the claims they support.\n"
     "- If the context only partly answers, answer the supported part and say what is missing.\n"
     "- Never invent APIs, parameters or code that are not in the context.\n"
     "- Query type is '{query_type}': for how-to give short steps/code; for conceptual be concise; "
     "for troubleshooting give likely cause then fix.\n{strict}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer is fully supported by the provided context. "
     "Answer 'yes' only if every factual claim in the answer can be found in or directly inferred "
     "from the context. Otherwise 'no'."),
    ("human", "Context:\n{context}\n\nAnswer:\n{generation}"),
])


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def get_llm():
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.LLM_MODEL, temperature=0, api_key=config.GROQ_API_KEY)


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata.get('source')})\n{d.page_content}" for i, d in enumerate(docs, 1))


def build_sources(docs: List[Document], generation: str) -> List[dict]:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", generation)}
    sources = [
        {
            "id": i,
            "source": d.metadata.get("source"),
            "section": d.metadata.get("section") or None,
            "chunk_id": d.metadata.get("chunk_id"),
            "snippet": d.page_content[:300],
            "cited": i in cited,
        }
        for i, d in enumerate(docs, 1)
    ]
    return sources


# ---------------------------------------------------------------- nodes
def analyze_query(state: GraphState) -> dict:
    question = state["question"]
    try:
        r = (ANALYZE_PROMPT | get_llm().with_structured_output(QueryAnalysis)).invoke({"question": question})
        query, qtype = r.rewritten_query, r.query_type
    except RuntimeError:
        raise
    except Exception as e:  # LLM hiccup must not kill the request
        log.warning("analyze_query failed (%s); using raw question", e)
        query, qtype = question, "conceptual"
    return {"query": query, "query_type": qtype, "retries": 0, "gen_attempts": 0, "grounded": None}


def retrieve(state: GraphState) -> dict:
    from app.vectorstore import get_vectorstore

    results = get_vectorstore().similarity_search_with_score(state["query"], k=config.TOP_K)
    docs = []
    for doc, distance in results:
        doc.metadata["score"] = round(1 - float(distance), 4)  # cosine similarity
        docs.append(doc)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"documents": [], "found_answer": False}
    grader = GRADE_PROMPT | get_llm().with_structured_output(GradeDocument)
    inputs = [{"question": state["question"], "document": d.page_content} for d in docs]
    grades = grader.batch(inputs, config={"max_concurrency": 4}, return_exceptions=True)

    kept = []
    for doc, g in zip(docs, grades):
        if isinstance(g, Exception):          # fail open: keep chunk if grader errored
            log.warning("grader error on %s: %s", doc.metadata.get("chunk_id"), g)
            kept.append(doc)
        elif g.binary_score == "yes":
            kept.append(doc)
    return {"documents": kept, "found_answer": bool(kept)}


def rewrite_query(state: GraphState) -> dict:
    new_query = (REWRITE_PROMPT | get_llm()).invoke(
        {"question": state["question"], "query": state["query"]}
    ).content.strip()
    return {"query": new_query or state["question"], "retries": state.get("retries", 0) + 1}


def generate(state: GraphState) -> dict:
    attempts = state.get("gen_attempts", 0)
    strict = (
        "\nIMPORTANT: a previous draft contained unsupported claims. Stick strictly to what the context says."
        if attempts > 0 else ""
    )
    docs = state["documents"]
    text = (GENERATE_PROMPT | get_llm()).invoke({
        "context": format_context(docs),
        "question": state["question"],
        "query_type": state.get("query_type", "conceptual"),
        "strict": strict,
    }).content
    return {"generation": text, "sources": build_sources(docs, text), "gen_attempts": attempts + 1}


def check_hallucination(state: GraphState) -> dict:
    try:
        g = (HALLUCINATION_PROMPT | get_llm().with_structured_output(GradeGrounded)).invoke({
            "context": format_context(state["documents"]),
            "generation": state["generation"],
        })
        return {"grounded": g.binary_score == "yes"}
    except Exception as e:
        log.warning("hallucination check failed: %s", e)
        return {"grounded": None}   # unknown


def no_answer(state: GraphState) -> dict:
    return {"generation": NO_ANSWER_TEXT, "sources": [], "grounded": None}


# ---------------------------------------------------------------- routing (conditional edges)
def route_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "no_answer"]:
    if state.get("found_answer"):
        return "generate"
    if state.get("retries", 0) < config.MAX_RETRIES:
        return "rewrite_query"
    return "no_answer"


def route_after_hallucination(state: GraphState) -> Literal["generate", "__end__"]:
    if state.get("grounded") is False and state.get("gen_attempts", 0) < config.MAX_GEN_ATTEMPTS:
        return "generate"
    return END


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("analyze_query", analyze_query)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("check_hallucination", check_hallucination)
    g.add_node("no_answer", no_answer)

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate", "rewrite_query": "rewrite_query", "no_answer": "no_answer"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_hallucination")
    g.add_conditional_edges("check_hallucination", route_after_hallucination,
                            {"generate": "generate", END: END})
    g.add_edge("no_answer", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
