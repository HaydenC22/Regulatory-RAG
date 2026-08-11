"""A tiny synthetic fixture corpus (not real MAS text) used by integration tests
so they're fast, deterministic, and independent of the real ingestion pipeline."""

import hashlib

EMBEDDING_DIM = 768

DOCUMENTS = [
    {
        "doc_id": "FIXDOC1",
        "title": "Fixture Notice One",
        "doc_type": "notice",
        "notice_number": "FX01",
        "effective_date": "2024-01-01",
        "last_revised_date": None,
        "source_url": "https://example.com/fixdoc1.pdf",
        "sha256": "fixture-sha-1",
    },
    {
        "doc_id": "FIXDOC2",
        "title": "Fixture Guideline Two",
        "doc_type": "guideline",
        "notice_number": None,
        "effective_date": "2023-06-01",
        "last_revised_date": None,
        "source_url": "https://example.com/fixdoc2.pdf",
        "sha256": "fixture-sha-2",
    },
]

CHUNKS = [
    {
        "chunk_id": "fix-1-1",
        "doc_id": "FIXDOC1",
        "section_heading": "Customer Due Diligence",
        "paragraph_id": "5.2",
        "page_number": 3,
        "chunk_index": 0,
        "token_count": 12,
        "text": "A regulated entity must perform customer due diligence before onboarding any new customer.",
    },
    {
        "chunk_id": "fix-1-2",
        "doc_id": "FIXDOC1",
        "section_heading": "Record Keeping",
        "paragraph_id": "7.1",
        "page_number": 5,
        "chunk_index": 1,
        "token_count": 10,
        "text": "Transaction records must be kept for a minimum of five years.",
    },
    {
        "chunk_id": "fix-2-1",
        "doc_id": "FIXDOC2",
        "section_heading": "Licensing",
        "paragraph_id": "2.1",
        "page_number": 1,
        "chunk_index": 0,
        "token_count": 11,
        "text": "A licence is required to provide cross-border money transfer services.",
    },
]


def deterministic_embedding(text: str) -> list[float]:
    """A cheap, deterministic stand-in for a real embedding — good enough for
    exercising the retrieval/API machinery without loading a real model in tests."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [(digest[i % len(digest)] - 128) / 128 for i in range(EMBEDDING_DIM)]
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]
