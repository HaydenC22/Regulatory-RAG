from services.ingestion.chunk import MAX_CHUNK_TOKENS, DocMeta, chunk_document, count_tokens
from services.ingestion.parse import SectionRecord

DOC_META = DocMeta(
    doc_id="TESTDOC",
    doc_title="Test Notice",
    doc_type="notice",
    notice_number="TN01",
    effective_date=None,
    last_revised_date=None,
    source_url="https://example.com/testdoc.pdf",
)


def make_records(paragraphs: list[tuple[str, str]]) -> list[SectionRecord]:
    """paragraphs: list of (paragraph_id, text)."""
    return [
        SectionRecord(page=1, heading="Test Section", paragraph_id=pid, text=text) for pid, text in paragraphs
    ]


def test_short_paragraphs_produce_correct_paragraph_ids():
    records = make_records([("1.1", "Short paragraph one."), ("1.2", "Short paragraph two.")])
    chunks = chunk_document(records, DOC_META)

    assert len(chunks) >= 1
    all_text = " ".join(text for _, text in chunks)
    assert "Short paragraph one." in all_text
    assert "Short paragraph two." in all_text
    # first chunk's metadata should carry the first paragraph's id, not be blank
    assert chunks[0][0].paragraph_id == "1.1"


def test_no_chunk_exceeds_token_limit_for_normal_paragraphs():
    # Many small paragraphs that individually stay well under the limit.
    records = make_records(
        [(f"1.{i}", f"This is paragraph number {i} with some content.") for i in range(50)]
    )
    chunks = chunk_document(records, DOC_META)

    for metadata, text in chunks:
        assert count_tokens(text) <= MAX_CHUNK_TOKENS + 50  # small slack for the prepended heading prefix


def test_oversized_single_paragraph_is_split_with_overlap():
    huge_text = "word " * 1000  # far exceeds MAX_CHUNK_TOKENS on its own
    records = make_records([("9.1", huge_text)])
    chunks = chunk_document(records, DOC_META)

    assert len(chunks) > 1, "an oversized single paragraph must be split into multiple chunks"
    for metadata, text in chunks:
        assert metadata.paragraph_id == "9.1"


def _text_with_token_count(word: str, target_tokens: int) -> str:
    """Builds text whose token count is close to (but not exceeding) target_tokens,
    measured via the real tokenizer rather than assumed word/token ratios."""
    words = []
    text = ""
    while count_tokens(text) < target_tokens:
        words.append(word)
        text = " ".join(words)
    words.pop()  # step back under the target
    return " ".join(words)


def test_paragraph_never_split_when_under_limit_even_if_buffer_would_overflow():
    # Two paragraphs that together exceed the limit but neither alone does —
    # the chunker must flush the buffer rather than splitting a paragraph.
    p1 = _text_with_token_count("alpha", 300)
    p2 = _text_with_token_count("beta", 300)
    assert count_tokens(p1) < MAX_CHUNK_TOKENS
    assert count_tokens(p2) < MAX_CHUNK_TOKENS

    records = make_records([("2.1", p1), ("2.2", p2)])
    chunks = chunk_document(records, DOC_META)

    assert len(chunks) == 2
    assert "alpha" in chunks[0][1] and "beta" not in chunks[0][1]
    assert "beta" in chunks[1][1] and "alpha" not in chunks[1][1]


def test_empty_records_produce_no_chunks():
    records = make_records([("1.1", "   "), ("1.2", "")])
    chunks = chunk_document(records, DOC_META)
    assert chunks == []
