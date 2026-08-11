from fastapi.testclient import TestClient

from services.agent.checklist import _ItemDraft
from services.agent.tools import ActivityClassification
from services.schemas import Citation


def test_checklist_endpoint_flags_low_confidence_items_for_manual_review(
    db_url, patched_embeddings, patched_rerank, patched_bm25, monkeypatch
):
    def fake_classify(description: str) -> ActivityClassification:
        return ActivityClassification(
            activities=["digital_payment_token_service"],
            reasoning="fixture: mentions digital tokens",
        )

    def fake_item_draft(prompt, schema, tier="cheap"):
        assert schema is _ItemDraft
        # First requirement gets a confident, well-cited answer; subsequent ones
        # deliberately cite nothing so the orchestration must mark them uncertain.
        if "DTSP licensing requirement" in prompt:
            return _ItemDraft(
                applicable="yes",
                rationale="Fixture: licensing applies to DPT services.",
                citations=[
                    Citation(
                        doc_id="FIXDOC2",
                        paragraph_id="2.1",
                        quote="A licence is required to provide cross-border money transfer services",
                    )
                ],
                confidence="high",
            )
        return _ItemDraft(
            applicable="uncertain",
            rationale="Fixture: insufficient context.",
            citations=[],
            confidence="low",
        )

    monkeypatch.setattr("services.agent.checklist.classify_business_activity", fake_classify)
    monkeypatch.setattr("services.agent.checklist.generate_structured", fake_item_draft)

    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/checklist",
        json={"business_description": "We operate a cross-border stablecoin payment service from Singapore."},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["classified_activities"] == ["digital_payment_token_service"]
    assert len(data["items"]) == 3  # ACTIVITY_REQUIREMENTS["digital_payment_token_service"] has 3 entries

    applicable_values = {item["applicable"] for item in data["items"]}
    assert "yes" in applicable_values
    assert "uncertain" in applicable_values

    yes_item = next(item for item in data["items"] if item["applicable"] == "yes")
    assert yes_item["citations"][0]["doc_id"] == "FIXDOC2"

    assert len(data["flagged_for_manual_review"]) >= 1
    assert data["overall_confidence"] == "low"


def test_checklist_endpoint_no_activities_detected_returns_empty_items(
    db_url, patched_embeddings, patched_rerank, patched_bm25, monkeypatch
):
    def fake_classify(description: str) -> ActivityClassification:
        return ActivityClassification(activities=[], reasoning="fixture: no regulated activity mentioned")

    monkeypatch.setattr("services.agent.checklist.classify_business_activity", fake_classify)

    from api.main import app

    client = TestClient(app)
    response = client.post("/api/v1/checklist", json={"business_description": "We sell artisanal candles."})

    assert response.status_code == 200
    data = response.json()
    assert data["classified_activities"] == []
    assert data["items"] == []
    assert data["overall_confidence"] == "high"
