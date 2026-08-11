from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse
from services.agent.qa import answer_question

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = answer_question(request.question, config="final")
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        retrieved_chunks=[c.metadata for c in result.retrieved_chunks],
        latency_ms=result.latency_ms,
        refused=result.refused,
    )
