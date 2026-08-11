"""Programmatic citation enforcement: every Citation the LLM returns is checked
against the chunks actually retrieved for that query, not the whole corpus.
This is what makes "an uncited answer is a bug" a testable guarantee rather than
a prompting aspiration."""

from dataclasses import dataclass

from rapidfuzz import fuzz

from services.schemas import Citation, RetrievedChunk

QUOTE_MATCH_THRESHOLD = 85.0


@dataclass
class CitationCheck:
    citation: Citation
    valid: bool
    reason: str | None = None


def _find_chunk(citation: Citation, retrieved: list[RetrievedChunk]) -> RetrievedChunk | None:
    for chunk in retrieved:
        if chunk.metadata.doc_id == citation.doc_id and chunk.metadata.paragraph_id == citation.paragraph_id:
            return chunk
    return None


def validate_citation(citation: Citation, retrieved: list[RetrievedChunk]) -> CitationCheck:
    chunk = _find_chunk(citation, retrieved)
    if chunk is None:
        return CitationCheck(citation, valid=False, reason="doc_id/paragraph_id not among retrieved chunks")

    similarity = fuzz.partial_ratio(citation.quote.lower(), chunk.text.lower())
    if similarity < QUOTE_MATCH_THRESHOLD:
        return CitationCheck(
            citation,
            valid=False,
            reason=f"quote fuzzy-match {similarity:.0f} < {QUOTE_MATCH_THRESHOLD}",
        )

    return CitationCheck(citation, valid=True)


def validate_all(citations: list[Citation], retrieved: list[RetrievedChunk]) -> list[CitationCheck]:
    return [validate_citation(c, retrieved) for c in citations]


def all_valid(checks: list[CitationCheck]) -> bool:
    return all(c.valid for c in checks)


def citation_validity_rate(checks: list[CitationCheck]) -> float:
    if not checks:
        return 0.0
    return sum(1 for c in checks if c.valid) / len(checks)
