"""Tools available to the compliance-checklist agent. Exposed both as plain
callables (used by the deterministic checklist orchestration in checklist.py)
and as LangChain @tool wrappers (for interactive/debugging use via an agent
executor, e.g. in a notebook)."""

from langchain_core.tools import tool
from pydantic import BaseModel

from services.llm import generate_structured
from services.retrieval.pipeline import retrieve
from services.schemas import ChunkMetadata, RetrievedChunk

REGULATED_ACTIVITIES = [
    "digital_payment_token_service",
    "cross_border_money_transfer_service",
    "domestic_money_transfer_service",
    "e_money_issuance",
    "merchant_acquisition_service",
    "account_issuance_service",
    "money_changing_service",
]


class ActivityClassification(BaseModel):
    activities: list[str]
    reasoning: str


def search_regulations(query: str) -> list[RetrievedChunk]:
    """Hybrid search + rerank over the MAS corpus, top 5 passages for `query`."""
    return retrieve(query, config="final")


def get_clause(doc_id: str, paragraph_id: str, retrieved: list[RetrievedChunk]) -> ChunkMetadata | None:
    """Exact lookup of a specific clause among already-retrieved chunks, for
    cross-referencing a citation."""
    for chunk in retrieved:
        if chunk.metadata.doc_id == doc_id and chunk.metadata.paragraph_id == paragraph_id:
            return chunk.metadata
    return None


def classify_business_activity(description: str) -> ActivityClassification:
    """Classifies a free-text business description into Payment Services Act
    regulated-activity categories."""
    prompt = (
        "Classify this FinTech business description into zero or more of these "
        f"Payment Services Act regulated-activity categories: {REGULATED_ACTIVITIES}\n\n"
        f'Business description: "{description}"\n\n'
        "Return only categories genuinely implied by the description — do not guess "
        "categories not supported by the text."
    )
    return generate_structured(prompt, ActivityClassification, tier="cheap")


search_regulations_tool = tool(search_regulations)
classify_business_activity_tool = tool(classify_business_activity)
