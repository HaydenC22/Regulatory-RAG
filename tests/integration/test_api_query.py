from fastapi.testclient import TestClient

from services.agent.guardrails import ScopeClassification
from services.schemas import Citation, GroundedAnswer


def _fake_generate_structured(prompt, schema, tier="cheap"):
    if schema is ScopeClassification:
        return ScopeClassification(in_scope=True, reason="fixture: in scope")
    if schema is GroundedAnswer:
        return GroundedAnswer(
            answer="A regulated entity must perform customer due diligence before onboarding.",
            citations=[
                Citation(
                    doc_id="FIXDOC1",
                    paragraph_id="5.2",
                    quote="A regulated entity must perform customer due diligence",
                )
            ],
            confidence="high",
        )
    raise AssertionError(f"Unexpected schema requested in mock: {schema}")


def test_query_endpoint_returns_grounded_answer_with_valid_citations(
    db_url, patched_embeddings, patched_rerank, patched_bm25, monkeypatch
):
    monkeypatch.setattr("services.llm.generate_structured", _fake_generate_structured)
    monkeypatch.setattr("services.agent.guardrails.generate_structured", _fake_generate_structured)

    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/query", json={"question": "What is required before onboarding a customer?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is False
    assert data["confidence"] == "high"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_id"] == "FIXDOC1"
    assert data["citations"][0]["paragraph_id"] == "5.2"
    assert data["latency_ms"] > 0
    assert any(c["doc_id"] == "FIXDOC1" for c in data["retrieved_chunks"])


def test_query_endpoint_refuses_when_out_of_scope(
    db_url, patched_embeddings, patched_rerank, patched_bm25, monkeypatch
):
    def refuse_scope(prompt, schema, tier="cheap"):
        if schema is ScopeClassification:
            return ScopeClassification(in_scope=False, reason="fixture: out of scope")
        raise AssertionError("GroundedAnswer should not be requested once scope check refuses")

    monkeypatch.setattr("services.llm.generate_structured", refuse_scope)
    monkeypatch.setattr("services.agent.guardrails.generate_structured", refuse_scope)

    from api.main import app

    client = TestClient(app)
    response = client.post("/api/v1/query", json={"question": "What's the weather today?"})

    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert data["citations"] == []


def test_health_endpoint():
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
