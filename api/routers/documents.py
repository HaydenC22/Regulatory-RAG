from fastapi import APIRouter, HTTPException

from api.schemas import DocumentSummary, EvalLatestResponse
from services.db import get_connection
from services.schemas import ChunkMetadata

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, title, doc_type, notice_number, effective_date, source_url "
            "FROM documents ORDER BY doc_id"
        ).fetchall()
    return [DocumentSummary(**row) for row in rows]


@router.get("/documents/{doc_id}/clauses/{paragraph_id}", response_model=ChunkMetadata)
def get_clause(doc_id: str, paragraph_id: str) -> ChunkMetadata:
    with get_connection() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
        chunk = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = %s AND paragraph_id = %s LIMIT 1",
            (doc_id, paragraph_id),
        ).fetchone()
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"No clause {paragraph_id} in {doc_id}")

    return ChunkMetadata(
        chunk_id=chunk["chunk_id"],
        doc_id=doc_id,
        doc_title=doc["title"],
        doc_type=doc["doc_type"],
        notice_number=doc["notice_number"],
        section_heading=chunk["section_heading"],
        paragraph_id=chunk["paragraph_id"],
        page_number=chunk["page_number"],
        effective_date=doc["effective_date"],
        last_revised_date=doc["last_revised_date"],
        source_url=doc["source_url"],
        chunk_index=chunk["chunk_index"],
        token_count=chunk["token_count"],
    )


@router.get("/eval/latest", response_model=list[EvalLatestResponse])
def eval_latest() -> list[EvalLatestResponse]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (config_name) *
            FROM eval_runs
            ORDER BY config_name, generated_at DESC
            """
        ).fetchall()
    return [EvalLatestResponse(**row) for row in rows]
