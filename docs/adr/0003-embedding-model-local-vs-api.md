# ADR-0003: Embedding model — local bge-base-en-v1.5 vs paid API embeddings

## Status
Accepted

## Context
Chunk embeddings are needed for ~1,500-2,500 chunks at ingestion time, and a
query embedding is needed on every `/query` and `/checklist` request. The
realistic choices were a local `sentence-transformers` model (no API key, no
per-call cost) or a paid embedding API (e.g. OpenAI `text-embedding-3-small`
or Voyage).

## Decision
Use **`BAAI/bge-base-en-v1.5`** (local, via `sentence-transformers`) as the
default and only wired-up embedding provider (`EMBEDDING_PROVIDER=local`).

## Rationale
- **$0 marginal cost** for both ingestion (one-time, ~2,000 chunks) and every
  query at demo time — meaningful for a portfolio project with a public live
  demo and a tightly capped API budget (see the deployment section of the
  README).
- **CI needs no API key.** Unit and integration tests never call a real
  embedding API; a local model keeps that true without extra mocking effort
  for the embedding step specifically (embeddings are still stubbed in
  integration tests for speed — see `tests/integration/conftest.py` — but the
  production code path itself has no external dependency).
- **Reranking is the higher-leverage quality lever** in this pipeline (see
  ADR-0002): the cross-encoder rerank step operates on the fused candidate
  set regardless of which embedding model produced the dense half of that
  set, which reduces how much the embedding model choice matters to final
  answer quality.

## Trade-off accepted
`bge-base-en-v1.5` (768 dimensions) likely trails a strong paid embedding
model on some retrieval-quality benchmarks. If `docs/eval/report_final.md`
shows `context_precision` capped below an acceptable level even after hybrid
search and reranking are in place, the next lever to pull is switching
`EMBEDDING_PROVIDER` to a paid API — the `services/ingestion/embed.py`
interface (`embed_texts` / `embed_query`) is provider-agnostic so this is a
contained change, not a rearchitecture. This is intentionally listed as a
"what's next" item in the README rather than pre-built, to keep the MVP's
cost surface at $0 for embeddings during development.
