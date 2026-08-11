from datetime import date, datetime

from pydantic import BaseModel

from services.schemas import ChunkMetadata, Citation, ComplianceChecklist, Confidence


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Confidence
    retrieved_chunks: list[ChunkMetadata]
    latency_ms: float
    refused: bool = False


class ChecklistRequest(BaseModel):
    business_description: str


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    notice_number: str | None
    effective_date: date | None
    source_url: str


class EvalLatestResponse(BaseModel):
    config_name: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    citation_validity_pct: float | None
    refusal_precision: float | None
    refusal_recall: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    generated_at: datetime


ChecklistResponse = ComplianceChecklist
