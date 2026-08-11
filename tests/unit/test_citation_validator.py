from services.citation.validator import (
    all_valid,
    citation_validity_rate,
    validate_all,
    validate_citation,
)
from services.schemas import ChunkMetadata, Citation, RetrievedChunk


def make_chunk(doc_id: str, paragraph_id: str, text: str) -> RetrievedChunk:
    metadata = ChunkMetadata(
        chunk_id=f"{doc_id}-{paragraph_id}",
        doc_id=doc_id,
        doc_title=f"{doc_id} title",
        doc_type="notice",
        paragraph_id=paragraph_id,
        page_number=1,
        source_url="https://example.com",
        chunk_index=0,
        token_count=10,
    )
    return RetrievedChunk(metadata=metadata, text=text, score=1.0)


RETRIEVED = [
    make_chunk("PSN02", "5.2", "Customer due diligence must be performed before onboarding."),
    make_chunk("PSN01", "3.1", "Enhanced due diligence applies to high-risk customers."),
]


def test_valid_citation_passes():
    citation = Citation(doc_id="PSN02", paragraph_id="5.2", quote="Customer due diligence must be performed")
    check = validate_citation(citation, RETRIEVED)
    assert check.valid


def test_citation_with_wrong_paragraph_id_fails():
    citation = Citation(doc_id="PSN02", paragraph_id="5.3", quote="Customer due diligence must be performed")
    check = validate_citation(citation, RETRIEVED)
    assert not check.valid
    assert "not among retrieved" in check.reason


def test_citation_with_unsupported_quote_fails():
    citation = Citation(
        doc_id="PSN02", paragraph_id="5.2", quote="This text does not appear anywhere in the chunk"
    )
    check = validate_citation(citation, RETRIEVED)
    assert not check.valid
    assert "fuzzy-match" in check.reason


def test_citation_for_doc_not_in_retrieved_set_fails():
    citation = Citation(doc_id="PSN07", paragraph_id="1.1", quote="anything")
    check = validate_citation(citation, RETRIEVED)
    assert not check.valid


def test_all_valid_true_only_when_every_citation_passes():
    good = Citation(doc_id="PSN02", paragraph_id="5.2", quote="Customer due diligence must be performed")
    bad = Citation(doc_id="PSN02", paragraph_id="5.2", quote="nonexistent content")
    checks = validate_all([good], RETRIEVED)
    assert all_valid(checks)

    checks_mixed = validate_all([good, bad], RETRIEVED)
    assert not all_valid(checks_mixed)


def test_citation_validity_rate():
    good = Citation(doc_id="PSN02", paragraph_id="5.2", quote="Customer due diligence must be performed")
    bad = Citation(doc_id="PSN02", paragraph_id="5.2", quote="nonexistent content")
    checks = validate_all([good, bad], RETRIEVED)
    assert citation_validity_rate(checks) == 0.5
    assert citation_validity_rate([]) == 0.0


def test_citation_coerces_whole_number_float_paragraph_id_to_str():
    # Some LLM structured-output backends infer a numeric type for a
    # whole-number paragraph_id (e.g. Gemini returning 3.0) despite the
    # schema declaring str — this must coerce, not raise.
    citation = Citation(doc_id="PSN02", paragraph_id=3.0, quote="anything")
    assert citation.paragraph_id == "3"


def test_citation_coerces_int_paragraph_id_to_str():
    citation = Citation(doc_id="PSN02", paragraph_id=3, quote="anything")
    assert citation.paragraph_id == "3"


def test_citation_preserves_none_paragraph_id():
    citation = Citation(doc_id="PSN02", paragraph_id=None, quote="anything")
    assert citation.paragraph_id is None
