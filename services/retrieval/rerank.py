"""Cross-encoder reranking of a fused candidate set down to the final top-k
context passed to the LLM. CPU inference, sub-second for ~20 candidate pairs."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from services.schemas import RetrievedChunk

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(_MODEL_NAME)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)
    reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [chunk.model_copy(update={"score": float(score)}) for chunk, score in reranked[:top_k]]
