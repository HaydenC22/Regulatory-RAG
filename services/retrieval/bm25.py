"""In-process BM25 lexical index over chunk text. Rebuilt from Postgres on first
use and cached in the API process for the process lifetime — fine at the corpus's
scale (~1500-2500 chunks). Call `invalidate()` after re-ingesting."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from services.db import get_connection
from services.schemas import ChunkMetadata, RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Index:
    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: dict[str, str]
    metadata: dict[str, ChunkMetadata]


_index: _Index | None = None


def invalidate() -> None:
    global _index
    _index = None


def _build_index() -> _Index:
    with get_connection() as conn:
        doc_lookup = {row["doc_id"]: row for row in conn.execute("SELECT * FROM documents").fetchall()}
        rows = conn.execute("SELECT * FROM chunks").fetchall()

    chunk_ids, corpus_tokens, texts, metadata = [], [], {}, {}
    for row in rows:
        doc = doc_lookup[row["doc_id"]]
        chunk_ids.append(row["chunk_id"])
        corpus_tokens.append(_tokenize(row["text"]))
        texts[row["chunk_id"]] = row["text"]
        metadata[row["chunk_id"]] = ChunkMetadata(
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

    return _Index(bm25=BM25Okapi(corpus_tokens), chunk_ids=chunk_ids, texts=texts, metadata=metadata)


def bm25_search(query: str, top_k: int = 20) -> list[RetrievedChunk]:
    global _index
    if _index is None:
        _index = _build_index()

    scores = _index.bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(_index.chunk_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)[
        :top_k
    ]
    return [
        RetrievedChunk(metadata=_index.metadata[chunk_id], text=_index.texts[chunk_id], score=float(score))
        for chunk_id, score in ranked
        if score > 0
    ]
