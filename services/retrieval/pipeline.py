"""The two retrieval configs compared in the eval harness (docs/eval/report_*.md).

baseline: dense-only, top 5, no reranking — the starting point.
final:    hybrid RRF (BM25 + dense) + cross-encoder rerank, top 5 — see ADR-0002.
"""

from typing import Literal

from services.config import get_settings
from services.retrieval.dense import dense_search
from services.retrieval.hybrid import hybrid_search
from services.retrieval.rerank import rerank
from services.schemas import RetrievedChunk

RetrievalConfig = Literal["baseline", "final"]


def retrieve(query: str, config: RetrievalConfig = "final", top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or get_settings().retrieval_top_k

    if config == "baseline":
        return dense_search(query, top_k=top_k)

    candidates = hybrid_search(query, top_k=20)
    return rerank(query, candidates, top_k=top_k)
