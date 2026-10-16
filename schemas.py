from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How do I declare a request body in FastAPI?"])


class Source(BaseModel):
    id: int
    source: Optional[str]
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    snippet: str
    cited: bool


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[Source]
    query_type: Optional[str] = None
    rewritten_query: Optional[str] = None
    retries: int = 0
    grounded: Optional[bool] = Field(None, description="Hallucination-check result; null if not checked")


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=2000)


class DocumentInfo(BaseModel):
    source: str
    chunks: int
