# ADR-0002: Hybrid (BM25 + dense, RRF) retrieval vs dense-only

## Status
Accepted

## Context
Regulatory text is full of exact-match anchors that matter for correctness:
notice numbers ("PSN02"), dates ("30 June 2025"), section references
("§5.2.1"), and defined terms. Pure dense (embedding-based) retrieval is
known to under-retrieve on these kinds of exact tokens, since embedding
similarity optimizes for semantic closeness, not lexical exactness.

## Decision
Use **hybrid retrieval**: BM25 (lexical) and dense (pgvector cosine) results
are fused with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), then the
fused top-20 candidates are reranked with a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) down to the final top-5 context. This
is the `final` retrieval config; `dense-only, no rerank` is kept as the
`baseline` config purely for the eval comparison in `docs/eval/report_*.md`.

## Rationale
- BM25 recovers exact-match anchors (notice numbers, dates, defined terms)
  that dense embeddings can miss or under-rank.
- Dense retrieval recovers paraphrased / semantically-related questions that
  don't share vocabulary with the source text.
- RRF is a simple, parameter-light way to fuse two differently-scaled ranking
  signals (BM25 scores and cosine similarities aren't on the same scale) —
  it only needs rank position, not score calibration.
- The cross-encoder rerank step is a second, more expensive but more accurate
  pass over a small candidate set (20 items), which is a standard two-stage
  retrieval pattern that keeps the final context both relevant and diverse.

## Trade-off accepted
Hybrid adds real complexity: two indexes to maintain (BM25 rebuilt in-process,
pgvector persisted), a fusion step, and reranker inference latency. This is
justified empirically, not just asserted — see the `context_precision` delta
between `report_baseline.md` and `report_final.md` after running `make eval`.
If that delta turns out to be small in practice, dense-only + rerank (skipping
BM25) would be a reasonable simplification to fall back to.
