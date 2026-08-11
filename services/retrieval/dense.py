from services.db import get_connection
from services.ingestion.embed import embed_query
from services.schemas import ChunkMetadata, RetrievedChunk

_SELECT_COLUMNS = """
    chunk_id, doc_id, section_heading, paragraph_id, page_number, chunk_index,
    token_count, text, 1 - (embedding <=> %(query_vec)s::vector) AS score
"""


def _row_to_retrieved_chunk(row: dict, doc_lookup: dict[str, dict]) -> RetrievedChunk:
    doc = doc_lookup[row["doc_id"]]
    metadata = ChunkMetadata(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_title=doc["title"],
        doc_type=doc["doc_type"],
        notice_number=doc["notice_number"],
        section_heading=row["section_heading"],
        paragraph_id=row["paragraph_id"],
        page_number=row["page_number"],
        effective_date=doc["effective_date"],
        last_revised_date=doc["last_revised_date"],
        source_url=doc["source_url"],
        chunk_index=row["chunk_index"],
        token_count=row["token_count"],
    )
    return RetrievedChunk(metadata=metadata, text=row["text"], score=row["score"])


def _load_doc_lookup(conn) -> dict[str, dict]:
    rows = conn.execute("SELECT * FROM documents").fetchall()
    return {row["doc_id"]: row for row in rows}


def dense_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    query_vec = embed_query(query)
    with get_connection() as conn:
        doc_lookup = _load_doc_lookup(conn)
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM chunks "
            "ORDER BY embedding <=> %(query_vec)s::vector LIMIT %(top_k)s",
            {"query_vec": query_vec, "top_k": top_k},
        ).fetchall()
    return [_row_to_retrieved_chunk(row, doc_lookup) for row in rows]
