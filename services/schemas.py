"""Shared Pydantic models used across ingestion, retrieval, agent, citation and API layers."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocType = Literal["notice", "guideline", "act_excerpt", "principles", "whitepaper"]
Confidence = Literal["high", "medium", "low"]
Applicability = Literal["yes", "no", "uncertain"]


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    notice_number: str | None = None
    section_heading: str | None = None
    paragraph_id: str | None = None
    page_number: int
    effective_date: date | None = None
    last_revised_date: date | None = None
    source_url: str
    chunk_index: int
    token_count: int


class RetrievedChunk(BaseModel):
    metadata: ChunkMetadata
    text: str
    score: float


class Citation(BaseModel):
    doc_id: str
    paragraph_id: str | None = None
    quote: str = Field(description="Short verbatim span from the cited chunk supporting the answer")

    @field_validator("paragraph_id", mode="before")
    @classmethod
    def _coerce_paragraph_id_to_str(cls, value: object) -> str | None:
        """LLM structured output has been observed to infer a numeric type
        for a whole-number paragraph_id (e.g. 3.0) despite the schema
        declaring str — coerce defensively rather than raising, since which
        provider/model does this is provider-dependent (see ADR-0004) and
        not something a prompt tweak alone should be relied on to prevent."""
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, int | float):
            return str(value)
        return value


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence


class ChecklistItem(BaseModel):
    requirement: str
    applicable: Applicability
    rationale: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence


class ComplianceChecklist(BaseModel):
    business_description: str
    classified_activities: list[str]
    items: list[ChecklistItem]
    overall_confidence: Confidence
    flagged_for_manual_review: list[str] = Field(default_factory=list)
