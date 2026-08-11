"""Orchestrates a single grounded-answer request: guardrail gate -> retrieve ->
generate structured GroundedAnswer -> validate citations -> bounded retry ->
refuse. This loop is what the "% of answers with valid citations" metric
measures, independently of RAGAS's faithfulness score."""

import time
from dataclasses import dataclass, field

from services.agent.guardrails import REFUSAL_MESSAGE, should_refuse
from services.citation.validator import all_valid, validate_all
from services.retrieval.pipeline import RetrievalConfig, retrieve
from services.schemas import Citation, GroundedAnswer, RetrievedChunk

MAX_RETRIES = 1


@dataclass
class QueryResult:
    answer: str
    citations: list[Citation]
    confidence: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    refused: bool = False


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """doc_id and paragraph_id are tagged as separate explicit fields (not a
    combined bracket like "[PSN02 §5]") because a combined tag was observed to
    make the model copy the whole tag into the Citation.doc_id field and fall
    back to a literal "n/a" string for paragraph_id instead of splitting the
    two — which made every citation fail validation regardless of how
    grounded the answer actually was.

    paragraph_id is quoted (paragraph_id="5") rather than left bare
    (paragraph_id=5), matching doc_id's quoting — an earlier unquoted version
    led Gemini's structured output to infer a numeric type for
    Citation.paragraph_id (e.g. 3.0), which fails Pydantic's `str | None`
    validation even though the schema is correct."""
    passages = []
    for c in chunks:
        paragraph_id = f'"{c.metadata.paragraph_id}"' if c.metadata.paragraph_id else "null"
        passages.append(f'[doc_id="{c.metadata.doc_id}" paragraph_id={paragraph_id}]\n{c.text}')
    return "\n\n".join(passages)


def _build_prompt(question: str, chunks: list[RetrievedChunk], invalid_note: str | None = None) -> str:
    context = _format_context(chunks)
    retry_note = (
        f"\n\nYour previous answer cited something not present in the context below "
        f"({invalid_note}). For each citation, set Citation.doc_id to EXACTLY the "
        'doc_id value shown (e.g. "PSN02", never combined with the paragraph_id) '
        'and Citation.paragraph_id to EXACTLY the paragraph_id value shown (e.g. "5", '
        "or null if the tag says paragraph_id=null — never the string 'n/a'). The quote "
        "must be a verbatim substring of that passage's text."
        if invalid_note
        else ""
    )
    return (
        "Answer the question using ONLY the context passages below. Every claim must "
        "be backed by a citation. Each passage is tagged with its doc_id and "
        'paragraph_id as separate fields, e.g. [doc_id="PSN02" paragraph_id=5] — '
        "copy those two values into the Citation.doc_id and Citation.paragraph_id "
        "fields exactly and separately (never combine them into one string, and use "
        "null, not the string 'n/a', when paragraph_id is null).\n\n"
        "Citation.quote must be SHORT (one sentence, ideally under 25 words) and copied "
        "CHARACTER-FOR-CHARACTER from the passage — never paraphrase, summarize, "
        "reword, or concatenate multiple sentences into the quote. Pick the single "
        "sentence or clause that most directly supports the claim and copy exactly "
        "that span verbatim. If no exact sentence in the context supports a claim, "
        "leave that claim out rather than inventing or loosely quoting a citation.\n\n"
        "If the context doesn't support an answer at all, say so and set confidence "
        "to 'low'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
        f"{retry_note}"
    )


def answer_question(question: str, config: RetrievalConfig = "final") -> QueryResult:
    from services.llm import generate_structured

    start = time.perf_counter()
    retrieved = retrieve(question, config=config)

    refuse, reason = should_refuse(question, retrieved)
    if refuse:
        return QueryResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            confidence="low",
            retrieved_chunks=retrieved,
            latency_ms=(time.perf_counter() - start) * 1000,
            refused=True,
        )

    invalid_note = None
    for attempt in range(MAX_RETRIES + 1):
        prompt = _build_prompt(question, retrieved, invalid_note)
        grounded = generate_structured(prompt, GroundedAnswer, tier="cheap")
        checks = validate_all(grounded.citations, retrieved)

        if all_valid(checks) and grounded.citations:
            return QueryResult(
                answer=grounded.answer,
                citations=grounded.citations,
                confidence=grounded.confidence,
                retrieved_chunks=retrieved,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        invalid_note = "; ".join(c.reason for c in checks if not c.valid) or "no citations provided"

    return QueryResult(
        answer=REFUSAL_MESSAGE,
        citations=[],
        confidence="low",
        retrieved_chunks=retrieved,
        latency_ms=(time.perf_counter() - start) * 1000,
        refused=True,
    )
