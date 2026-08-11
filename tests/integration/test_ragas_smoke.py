"""Smoke-tests that the RAGAS harness executes end-to-end and produces a
report with the expected shape. Does NOT assert on score thresholds — RAGAS
scores against a live LLM judge are non-deterministic and cost money, so CI
only checks the harness doesn't break, with both the answer-generation LLM
and the RAGAS judge call fully mocked (no LLM API key needed)."""

import pandas as pd

from eval.ragas_harness import GoldRecord, run_eval
from services.agent.guardrails import ScopeClassification
from services.schemas import Citation, GroundedAnswer

GOLD_SUBSET = [
    GoldRecord(
        id="F01",
        question="What must be done before onboarding a new customer?",
        question_type="factual",
        gold_answer="Customer due diligence must be performed before onboarding.",
        gold_citations=[{"doc_id": "FIXDOC1", "paragraph_id": "5.2"}],
        is_answerable=True,
    ),
    GoldRecord(
        id="F02",
        question="How long must transaction records be kept?",
        question_type="factual",
        gold_answer="A minimum of five years.",
        gold_citations=[{"doc_id": "FIXDOC1", "paragraph_id": "7.1"}],
        is_answerable=True,
    ),
    GoldRecord(
        id="O01",
        question="What's the weather like today?",
        question_type="out_of_scope",
        gold_answer="Out of scope.",
        gold_citations=[],
        is_answerable=False,
    ),
]


def _fake_generate_structured(prompt, schema, tier="cheap"):
    if schema is ScopeClassification:
        in_scope = "weather" not in prompt.lower()
        return ScopeClassification(in_scope=in_scope, reason="fixture")
    if schema is GroundedAnswer:
        if "customer" in prompt.lower() and "due diligence" in prompt.lower():
            return GroundedAnswer(
                answer="Customer due diligence must be performed before onboarding.",
                citations=[Citation(doc_id="FIXDOC1", paragraph_id="5.2", quote="customer due diligence")],
                confidence="high",
            )
        return GroundedAnswer(
            answer="Transaction records must be kept for a minimum of five years.",
            citations=[
                Citation(doc_id="FIXDOC1", paragraph_id="7.1", quote="kept for a minimum of five years")
            ],
            confidence="high",
        )
    raise AssertionError(f"Unexpected schema: {schema}")


class _FakeRagasReport:
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([{"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8}])


def test_ragas_harness_runs_end_to_end_and_produces_expected_report_shape(
    db_url, patched_embeddings, patched_rerank, patched_bm25, monkeypatch
):
    monkeypatch.setattr("services.llm.generate_structured", _fake_generate_structured)
    monkeypatch.setattr("services.agent.guardrails.generate_structured", _fake_generate_structured)
    monkeypatch.setattr("eval.ragas_harness._judge_model", lambda: object())
    monkeypatch.setattr("eval.ragas_harness.evaluate", lambda dataset, metrics, llm: _FakeRagasReport())

    result = run_eval(GOLD_SUBSET, config="final")

    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.85
    assert result.context_precision == 0.8
    assert 0.0 <= result.citation_validity_pct <= 100.0
    assert 0.0 <= result.refusal_precision <= 1.0
    assert 0.0 <= result.refusal_recall <= 1.0
    assert result.latency_p50_ms >= 0.0
    assert len(result.per_question) == 3

    # The out-of-scope question should have been refused, contributing to refusal recall.
    out_of_scope_row = next(row for row in result.per_question if row["id"] == "O01")
    assert out_of_scope_row["refused"] is True
    assert out_of_scope_row["expected_refusal"] is True
