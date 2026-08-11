CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    doc_type         TEXT NOT NULL CHECK (doc_type IN ('notice', 'guideline', 'act_excerpt', 'principles', 'whitepaper')),
    notice_number    TEXT,
    effective_date   DATE,
    last_revised_date DATE,
    source_url       TEXT NOT NULL,
    sha256           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_heading  TEXT,
    paragraph_id     TEXT,
    page_number      INT,
    chunk_index      INT NOT NULL,
    token_count      INT NOT NULL,
    text             TEXT NOT NULL,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding        vector(768)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- Records one row per baseline/final eval run, backing GET /api/v1/eval/latest
CREATE TABLE IF NOT EXISTS eval_runs (
    id                    SERIAL PRIMARY KEY,
    config_name           TEXT NOT NULL CHECK (config_name IN ('baseline', 'final')),
    faithfulness          FLOAT,
    answer_relevancy      FLOAT,
    context_precision     FLOAT,
    citation_validity_pct FLOAT,
    refusal_precision     FLOAT,
    refusal_recall        FLOAT,
    latency_p50_ms        FLOAT,
    latency_p95_ms        FLOAT,
    generated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
