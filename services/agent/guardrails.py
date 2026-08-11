"""Refusal / hallucination-mitigation guardrails.

Two independent gates decide whether a question gets a full retrieval+generation
cycle or an immediate documented refusal:
  1. Scope pre-check — a cheap structured-output classification of whether the
     question is plausibly covered by the MAS corpus this system indexes.
  2. Retrieval-confidence gate — even if scope-plausible, a top reranked score
     below RERANK_THRESHOLD means "no sufficient grounding" regardless of what
     the LLM would otherwise say.
"""

from pydantic import BaseModel

from services.config import get_settings
from services.llm import generate_structured
from services.schemas import RetrievedChunk

CORPUS_DESCRIPTION = (
    "Singapore MAS (Monetary Authority of Singapore) financial regulation: "
    "Technology Risk Management Guidelines, AML/CFT Notices PSN01/PSN02/PSN07, "
    "the FEAT principles for responsible AI/data analytics, DTSP licensing "
    "guidelines, the Payment Services Act 2019, and the Project MindForge "
    "generative-AI-for-banks whitepaper."
)

REFUSAL_MESSAGE = (
    "I don't have sufficient grounded evidence in the indexed MAS regulatory "
    "corpus to answer this confidently. This system covers Singapore MAS "
    "financial regulation only (see corpus description) — for anything outside "
    "that scope, or where retrieval confidence is too low, I refuse rather than "
    "guess."
)


class ScopeClassification(BaseModel):
    in_scope: bool
    reason: str


def classify_scope(question: str) -> ScopeClassification:
    prompt = (
        f"You are a scope classifier for a regulatory Q&A system that only indexes: "
        f"{CORPUS_DESCRIPTION}\n\n"
        f'Question: "{question}"\n\n'
        "Is this question plausibly answerable from that corpus? Answer in_scope=false "
        "for anything about non-Singapore regulators (e.g. EU MiCA, US SEC/FinCEN), "
        "general knowledge unrelated to financial regulation, or requests unrelated "
        "to MAS/Singapore financial compliance."
    )
    return generate_structured(prompt, ScopeClassification, tier="cheap")


def retrieval_confidence_gate(retrieved: list[RetrievedChunk]) -> bool:
    """Returns True if the top retrieved chunk clears the reranker-score threshold."""
    if not retrieved:
        return False
    threshold = get_settings().rerank_threshold
    return retrieved[0].score >= threshold


def should_refuse(question: str, retrieved: list[RetrievedChunk]) -> tuple[bool, str]:
    scope = classify_scope(question)
    if not scope.in_scope:
        return True, f"out of scope: {scope.reason}"
    if not retrieval_confidence_gate(retrieved):
        return True, "retrieval confidence below threshold"
    return False, ""
