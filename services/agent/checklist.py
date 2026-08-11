"""LangGraph orchestration for the business-description -> licensing/obligation
checklist flow. classify -> per-activity requirement generation -> assemble.

An item is marked "uncertain" (never guessed yes/no) whenever the reranker's top
score for its sub-query is below RERANK_THRESHOLD, or the LLM's own citation-
backed answer comes back with confidence="low" — see docs/adr and the ambiguous
bucket of docs/eval/gold_set.jsonl for how that threshold was calibrated."""

from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from services.agent.qa import _format_context
from services.agent.tools import classify_business_activity, search_regulations
from services.citation.validator import all_valid, validate_all
from services.config import get_settings
from services.llm import generate_structured
from services.schemas import Applicability, ChecklistItem, Citation, ComplianceChecklist, Confidence

ACTIVITY_REQUIREMENTS: dict[str, list[str]] = {
    "digital_payment_token_service": [
        "DTSP licensing requirement for a digital payment token service provider",
        "AML/CFT customer due diligence requirements for digital payment token service under Notice PSN02",
        "Compliance officer residency requirement for DTSP licensees",
    ],
    "cross_border_money_transfer_service": [
        "Licensing requirement for cross-border money transfer service under Payment Services Act",
        "AML/CFT requirements for cross-border money transfers under Notice PSN01",
    ],
    "domestic_money_transfer_service": [
        "Licensing requirement for domestic money transfer service under Payment Services Act",
    ],
    "e_money_issuance": [
        "Licensing and safeguarding requirements for e-money issuance under Payment Services Act",
    ],
    "merchant_acquisition_service": [
        "Licensing requirement for merchant acquisition service under Payment Services Act",
    ],
    "account_issuance_service": [
        "Licensing requirement for account issuance service under Payment Services Act",
    ],
    "money_changing_service": [
        "Licensing requirement for money-changing service under Payment Services Act",
    ],
}


class _ItemDraft(BaseModel):
    applicable: Applicability
    rationale: str
    citations: list[Citation]
    confidence: Confidence


class ChecklistState(TypedDict):
    business_description: str
    classified_activities: list[str]
    items: list[ChecklistItem]
    flagged_for_manual_review: list[str]


def _classify_node(state: ChecklistState) -> ChecklistState:
    result = classify_business_activity(state["business_description"])
    return {**state, "classified_activities": result.activities}


def _generate_item(requirement: str) -> ChecklistItem:
    retrieved = search_regulations(requirement)
    threshold = get_settings().rerank_threshold

    context = _format_context(retrieved)
    prompt = (
        "Based ONLY on the context passages below, determine whether this compliance "
        f'requirement applies: "{requirement}"\n\n'
        "If the context is insufficient or conflicting, set applicable='uncertain' and "
        "confidence='low' rather than guessing. Each passage is tagged with its doc_id "
        'and paragraph_id as separate fields, e.g. [doc_id="PSN02" paragraph_id=5] — '
        "copy those two values into Citation.doc_id and Citation.paragraph_id exactly "
        "and separately (never combine them into one string, and use null, not the "
        "string 'n/a', when paragraph_id is null).\n\n"
        "Citation.quote must be SHORT (one sentence, ideally under 25 words) and copied "
        "CHARACTER-FOR-CHARACTER from the passage — never paraphrase, summarize, or "
        "concatenate multiple sentences. Pick the single sentence or clause that most "
        "directly supports the determination and copy exactly that span verbatim.\n\n"
        f"Context:\n{context}"
    )
    draft = generate_structured(prompt, _ItemDraft, tier="cheap")

    checks = validate_all(draft.citations, retrieved)
    applicable = draft.applicable
    confidence = draft.confidence
    if not all_valid(checks) or not retrieved or retrieved[0].score < threshold:
        applicable = "uncertain"
        confidence = "low"

    return ChecklistItem(
        requirement=requirement,
        applicable=applicable,
        rationale=draft.rationale,
        citations=[c.citation for c in checks if c.valid],
        confidence=confidence,
    )


def _items_node(state: ChecklistState) -> ChecklistState:
    items: list[ChecklistItem] = []
    flagged: list[str] = []
    for activity in state["classified_activities"]:
        for requirement in ACTIVITY_REQUIREMENTS.get(activity, []):
            item = _generate_item(requirement)
            items.append(item)
            if item.applicable == "uncertain" or item.confidence == "low":
                flagged.append(item.requirement)
    return {**state, "items": items, "flagged_for_manual_review": flagged}


def _build_graph():
    graph = StateGraph(ChecklistState)
    graph.add_node("classify", _classify_node)
    graph.add_node("generate_items", _items_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "generate_items")
    graph.add_edge("generate_items", END)
    return graph.compile()


_graph = None


def build_checklist(business_description: str) -> ComplianceChecklist:
    global _graph
    if _graph is None:
        _graph = _build_graph()

    result: ChecklistState = _graph.invoke(
        {
            "business_description": business_description,
            "classified_activities": [],
            "items": [],
            "flagged_for_manual_review": [],
        }
    )

    overall_confidence: Confidence = "high"
    if any(item.confidence == "low" for item in result["items"]):
        overall_confidence = "low"
    elif any(item.confidence == "medium" for item in result["items"]):
        overall_confidence = "medium"

    return ComplianceChecklist(
        business_description=business_description,
        classified_activities=result["classified_activities"],
        items=result["items"],
        overall_confidence=overall_confidence,
        flagged_for_manual_review=result["flagged_for_manual_review"],
    )
