"""Section/paragraph-aware chunker.

Chunk boundaries snap to the paragraph_id boundaries produced by parse.py: a
numbered paragraph is never split mid-sentence across chunks. Overlap is only
applied when a single paragraph exceeds MAX_CHUNK_TOKENS on its own. This is the
file whose correctness determines citation granularity and retrieval quality —
see docs/adr and eval/ for how chunk quality is measured.
"""

import hashlib
from dataclasses import dataclass
from datetime import date

import tiktoken

from services.ingestion.parse import SectionRecord
from services.schemas import ChunkMetadata

MAX_CHUNK_TOKENS = 400
TARGET_MIN_TOKENS = 200
OVERLAP_TOKENS = 60

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


@dataclass
class DocMeta:
    doc_id: str
    doc_title: str
    doc_type: str
    notice_number: str | None
    effective_date: date | None
    last_revised_date: date | None
    source_url: str


def _make_chunk_id(doc_id: str, paragraph_id: str | None, chunk_index: int) -> str:
    key = f"{doc_id}:{paragraph_id or 'none'}:{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _split_oversized(record: SectionRecord) -> list[str]:
    """Splits a single paragraph that exceeds MAX_CHUNK_TOKENS into overlapping
    token windows — the only case where overlap is applied."""
    tokens = _encoding.encode(record.text)
    if len(tokens) <= MAX_CHUNK_TOKENS:
        return [record.text]

    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + MAX_CHUNK_TOKENS, len(tokens))
        windows.append(_encoding.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS
    return windows


def _format_chunk_text(doc_title: str, heading: str | None, paragraph_id: str | None, body: str) -> str:
    prefix_parts = [doc_title]
    if paragraph_id:
        prefix_parts.append(f"§{paragraph_id}")
    if heading:
        prefix_parts.append(f"— {heading}")
    prefix = " ".join(prefix_parts)
    return f"{prefix}: {body}"


def chunk_document(records: list[SectionRecord], doc_meta: DocMeta) -> list[tuple[ChunkMetadata, str]]:
    """Returns a list of (metadata, chunk_text) pairs for a single document."""
    chunks: list[tuple[ChunkMetadata, str]] = []
    chunk_index = 0

    buffer: list[SectionRecord] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, chunk_index
        if not buffer:
            return
        first = buffer[0]
        body = " ".join(r.text for r in buffer)
        chunk_text = _format_chunk_text(doc_meta.doc_title, first.heading, first.paragraph_id, body)
        metadata = ChunkMetadata(
            chunk_id=_make_chunk_id(doc_meta.doc_id, first.paragraph_id, chunk_index),
            doc_id=doc_meta.doc_id,
            doc_title=doc_meta.doc_title,
            doc_type=doc_meta.doc_type,
            notice_number=doc_meta.notice_number,
            section_heading=first.heading,
            paragraph_id=first.paragraph_id,
            page_number=first.page,
            effective_date=doc_meta.effective_date,
            last_revised_date=doc_meta.last_revised_date,
            source_url=doc_meta.source_url,
            chunk_index=chunk_index,
            token_count=count_tokens(chunk_text),
        )
        chunks.append((metadata, chunk_text))
        chunk_index += 1
        buffer = []
        buffer_tokens = 0

    for record in records:
        if not record.text.strip():
            continue

        record_tokens = count_tokens(record.text)

        if record_tokens > MAX_CHUNK_TOKENS:
            flush()
            for window_text in _split_oversized(record):
                chunk_text = _format_chunk_text(
                    doc_meta.doc_title, record.heading, record.paragraph_id, window_text
                )
                metadata = ChunkMetadata(
                    chunk_id=_make_chunk_id(doc_meta.doc_id, record.paragraph_id, chunk_index),
                    doc_id=doc_meta.doc_id,
                    doc_title=doc_meta.doc_title,
                    doc_type=doc_meta.doc_type,
                    notice_number=doc_meta.notice_number,
                    section_heading=record.heading,
                    paragraph_id=record.paragraph_id,
                    page_number=record.page,
                    effective_date=doc_meta.effective_date,
                    last_revised_date=doc_meta.last_revised_date,
                    source_url=doc_meta.source_url,
                    chunk_index=chunk_index,
                    token_count=count_tokens(chunk_text),
                )
                chunks.append((metadata, chunk_text))
                chunk_index += 1
            continue

        if buffer and buffer_tokens + record_tokens > MAX_CHUNK_TOKENS:
            flush()

        buffer.append(record)
        buffer_tokens += record_tokens

    flush()
    return chunks
