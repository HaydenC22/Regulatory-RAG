"""Ingestion entrypoint: fetch -> parse -> chunk -> embed -> load into pgvector.
Also builds the in-process BM25 index cache used by services/retrieval/bm25.py.

Run via `make ingest` (wraps `python -m services.ingestion.load`).
"""

import json
import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from services.config import get_settings
from services.ingestion import fetch
from services.ingestion.chunk import DocMeta, chunk_document
from services.ingestion.embed import embed_texts
from services.ingestion.parse import SectionRecord, parse_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _doc_meta_from_manifest(doc: dict) -> DocMeta:
    return DocMeta(
        doc_id=doc["doc_id"],
        doc_title=doc["title"],
        doc_type=doc["doc_type"],
        notice_number=doc.get("notice_number"),
        effective_date=doc.get("effective_date"),
        last_revised_date=doc.get("last_revised_date"),
        source_url=doc.get("pdf_url") or doc.get("page_url"),
    )


def _upsert_document(conn: psycopg.Connection, doc: dict, sha256: str) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, title, doc_type, notice_number, effective_date,
                                last_revised_date, source_url, sha256)
        VALUES (%(doc_id)s, %(title)s, %(doc_type)s, %(notice_number)s, %(effective_date)s,
                %(last_revised_date)s, %(source_url)s, %(sha256)s)
        ON CONFLICT (doc_id) DO UPDATE SET
            title = EXCLUDED.title, sha256 = EXCLUDED.sha256,
            last_revised_date = EXCLUDED.last_revised_date
        """,
        {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "doc_type": doc["doc_type"],
            "notice_number": doc.get("notice_number"),
            "effective_date": doc.get("effective_date"),
            "last_revised_date": doc.get("last_revised_date"),
            "source_url": doc.get("pdf_url") or doc.get("page_url"),
            "sha256": sha256,
        },
    )


def _write_processed_json(processed_dir: Path, doc_id: str, records: list[SectionRecord]) -> None:
    path = processed_dir / f"{doc_id}.json"
    path.write_text(
        json.dumps([r.__dict__ for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ingest_document(
    conn: psycopg.Connection, doc: dict, raw_dir: Path, processed_dir: Path, lock: dict
) -> int:
    doc_id = doc["doc_id"]
    pdf_path = raw_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        logger.warning("Skipping %s: %s not found (fetch step may have failed).", doc_id, pdf_path)
        return 0

    records = parse_pdf(str(pdf_path))
    _write_processed_json(processed_dir, doc_id, records)

    doc_meta = _doc_meta_from_manifest(doc)
    chunk_pairs = chunk_document(records, doc_meta)
    if not chunk_pairs:
        logger.warning("No chunks produced for %s", doc_id)
        return 0

    texts = [text for _, text in chunk_pairs]
    vectors = embed_texts(texts)

    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    _upsert_document(conn, doc, lock.get(doc_id, {}).get("sha256", ""))

    with conn.cursor() as cur:
        for (metadata, text), vector in zip(chunk_pairs, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, section_heading, paragraph_id,
                                     page_number, chunk_index, token_count, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                (
                    metadata.chunk_id,
                    metadata.doc_id,
                    metadata.section_heading,
                    metadata.paragraph_id,
                    metadata.page_number,
                    metadata.chunk_index,
                    metadata.token_count,
                    text,
                    vector,
                ),
            )
    conn.commit()
    logger.info("Ingested %s: %d chunks", doc_id, len(chunk_pairs))
    return len(chunk_pairs)


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.corpus_raw_dir)
    processed_dir = Path(settings.corpus_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fetch.main()  # downloads any missing/updated PDFs into corpus/raw/ first

    documents = fetch.load_manifest(settings.corpus_manifest_path)
    lock_path = Path(settings.corpus_manifest_path).parent / "manifest.lock.yaml"
    lock = fetch.load_lock(lock_path)

    total = 0
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        register_vector(conn)
        for doc in documents:
            total += ingest_document(conn, doc, raw_dir, processed_dir, lock)

    logger.info("Ingestion complete: %d total chunks across %d documents.", total, len(documents))


if __name__ == "__main__":
    main()
