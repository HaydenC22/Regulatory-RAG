"""Reciprocal Rank Fusion of BM25 and dense retrieval results — see ADR-0002."""

from services.retrieval.bm25 import bm25_search
from services.retrieval.dense import dense_search
from services.schemas import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]], k: int = RRF_K) -> list[RetrievedChunk]:
    """Fuses multiple ranked lists of RetrievedChunk by chunk_id using RRF:
    score(chunk) = sum over lists of 1 / (k + rank), rank is 1-indexed."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk.metadata.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(chunk_id, chunk)

    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [by_id[chunk_id].model_copy(update={"score": score}) for chunk_id, score in fused]


def hybrid_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    dense_results = dense_search(query, top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)
    return reciprocal_rank_fusion([dense_results, bm25_results])[:top_k]
