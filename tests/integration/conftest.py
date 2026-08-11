"""Spins up a real Postgres+pgvector container (via Testcontainers) for
integration tests, seeds the small synthetic fixture corpus, and patches the
embedding functions to a deterministic stand-in so tests don't need to
download a real sentence-transformers model. All LLM calls are mocked in the
individual test modules — no LLM API key is needed to run this suite."""

from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from testcontainers.postgres import PostgresContainer

from tests.fixtures.mini_corpus import CHUNKS, DOCUMENTS, deterministic_embedding

INIT_SQL_PATH = Path(__file__).resolve().parent.parent.parent / "infra" / "db" / "init.sql"


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest.fixture()
def db_url(postgres_container, monkeypatch):
    url = postgres_container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(INIT_SQL_PATH.read_text(encoding="utf-8"))
        register_vector(conn)  # must run after CREATE EXTENSION vector (in init.sql)
        conn.execute("TRUNCATE chunks, documents, eval_runs CASCADE")

        for doc in DOCUMENTS:
            conn.execute(
                """
                INSERT INTO documents (doc_id, title, doc_type, notice_number, effective_date,
                                        last_revised_date, source_url, sha256)
                VALUES (%(doc_id)s, %(title)s, %(doc_type)s, %(notice_number)s, %(effective_date)s,
                        %(last_revised_date)s, %(source_url)s, %(sha256)s)
                """,
                doc,
            )
        for chunk in CHUNKS:
            embedding = deterministic_embedding(chunk["text"])
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, section_heading, paragraph_id, page_number,
                                     chunk_index, token_count, text, embedding)
                VALUES (%(chunk_id)s, %(doc_id)s, %(section_heading)s, %(paragraph_id)s, %(page_number)s,
                        %(chunk_index)s, %(token_count)s, %(text)s, %(embedding)s)
                """,
                {**chunk, "embedding": embedding},
            )

    monkeypatch.setenv("DATABASE_URL", url)

    from services.config import get_settings
    from services.db import get_pool

    get_settings.cache_clear()
    get_pool.cache_clear()

    yield url

    get_settings.cache_clear()
    get_pool.cache_clear()


@pytest.fixture()
def patched_embeddings(monkeypatch):
    # Patched at each import site (`from ... import embed_query`), not just the
    # source module, since the importing modules already bound the original names.
    fake_query = lambda text: deterministic_embedding(f"query: {text}")  # noqa: E731
    fake_texts = lambda texts: [deterministic_embedding(t) for t in texts]  # noqa: E731

    monkeypatch.setattr("services.ingestion.embed.embed_query", fake_query)
    monkeypatch.setattr("services.ingestion.embed.embed_texts", fake_texts)
    monkeypatch.setattr("services.retrieval.dense.embed_query", fake_query)


@pytest.fixture()
def patched_rerank(monkeypatch):
    """Avoids downloading the real cross-encoder model in CI: passes candidates
    through unchanged (already ranked by RRF), truncated to top_k."""

    def fake_rerank(query, candidates, top_k=5):
        return candidates[:top_k]

    monkeypatch.setattr("services.retrieval.pipeline.rerank", fake_rerank)


@pytest.fixture()
def patched_bm25(monkeypatch):
    """BM25 index is rebuilt lazily and cached at module scope; reset between tests."""
    import services.retrieval.bm25 as bm25_module

    bm25_module.invalidate()
    yield
    bm25_module.invalidate()
