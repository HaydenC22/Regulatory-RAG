# ADR-0001: Vector store — pgvector vs Chroma

## Status
Accepted

## Context
The retrieval layer needs a vector store for chunk embeddings, plus somewhere
to persist the document manifest, chunk metadata, and eval-run history. The
two realistic options for an MVP of this size (~1,500-2,500 chunks) were
pgvector (Postgres extension) and Chroma (embedded/standalone vector DB).

## Decision
Use **pgvector** on Postgres as the single data store for documents, chunks,
embeddings, and eval run history.

## Rationale
- **Single stateful service.** `docker compose up` only needs one database
  container instead of Postgres + Chroma, which matters directly for the
  "runs in 60 seconds" engineering standard.
- **One system of record.** The document manifest, chunk metadata, and
  `eval_runs` history already need a relational store; pgvector means one
  connection pool and one backup/restore story instead of two.
- **Hybrid search is SQL-native.** BM25-style lexical scoring (via
  `tsvector`/`GIN`) and pgvector's `<=>` cosine operator can both be reached
  from the same query layer, which mirrors how a real production system would
  likely be built rather than gluing together two separate stores.
- **Realistic free-tier deployment path.** Supabase and Neon both offer
  pgvector-enabled Postgres on their free tiers, which the live demo depends
  on (see the deployment section of the README).

## Trade-off accepted
Chroma has a faster local dev loop (zero schema/migration ceremony) and
tighter out-of-the-box LangChain/LlamaIndex retriever integration. For a
larger corpus or a system where the vector store scales independently of the
relational data, Chroma (or a dedicated vector DB like Qdrant/Pinecone) would
be the better call. At this project's scale, that convenience doesn't
outweigh the "one system of record" and deployment simplicity argument.
