from datetime import datetime

from pydantic import BaseModel, Field


class DocumentRead(BaseModel):
    id: str
    title: str
    filename: str
    sha256: str
    mime_type: str
    page_count: int
    chunk_count: int
    warning_count: int
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(min_length=4, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = None


class Citation(BaseModel):
    document_id: str
    document_title: str
    page: int
    chunk_id: str
    excerpt: str
    score: float
    untrusted_content: bool


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: str
    score: float
    citations: list[Citation]
    warnings: list[str]
    latency_ms: float


class MetricsResponse(BaseModel):
    documents: int
    chunks: int
    queries: int
    flagged_chunks: int
