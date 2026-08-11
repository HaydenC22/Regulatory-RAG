from unittest.mock import patch

from services.agent.guardrails import ScopeClassification, retrieval_confidence_gate, should_refuse
from services.schemas import ChunkMetadata, RetrievedChunk


def make_chunk(score: float) -> RetrievedChunk:
    metadata = ChunkMetadata(
        chunk_id="c1",
        doc_id="PSN02",
        doc_title="Doc",
        doc_type="notice",
        page_number=1,
        source_url="https://example.com",
        chunk_index=0,
        token_count=10,
    )
    return RetrievedChunk(metadata=metadata, text="text", score=score)


def test_confidence_gate_passes_above_threshold():
    with patch("services.agent.guardrails.get_settings") as mock_settings:
        mock_settings.return_value.rerank_threshold = 0.0
        assert retrieval_confidence_gate([make_chunk(0.5)]) is True


def test_confidence_gate_fails_below_threshold():
    with patch("services.agent.guardrails.get_settings") as mock_settings:
        mock_settings.return_value.rerank_threshold = 0.0
        assert retrieval_confidence_gate([make_chunk(-0.1)]) is False


def test_confidence_gate_exactly_at_threshold_passes():
    with patch("services.agent.guardrails.get_settings") as mock_settings:
        mock_settings.return_value.rerank_threshold = 0.5
        assert retrieval_confidence_gate([make_chunk(0.5)]) is True


def test_confidence_gate_fails_on_empty_retrieval():
    assert retrieval_confidence_gate([]) is False


def test_should_refuse_out_of_scope_short_circuits_before_confidence_check():
    with patch("services.agent.guardrails.classify_scope") as mock_classify:
        mock_classify.return_value = ScopeClassification(in_scope=False, reason="not about MAS regulation")
        refuse, reason = should_refuse("what's the weather today?", [make_chunk(0.9)])
        assert refuse is True
        assert "out of scope" in reason


def test_should_refuse_in_scope_but_low_confidence_still_refuses():
    with (
        patch("services.agent.guardrails.classify_scope") as mock_classify,
        patch("services.agent.guardrails.get_settings") as mock_settings,
    ):
        mock_classify.return_value = ScopeClassification(in_scope=True, reason="about PSN02")
        mock_settings.return_value.rerank_threshold = 0.0
        refuse, reason = should_refuse("What does PSN02 require?", [make_chunk(-1.0)])
        assert refuse is True
        assert "confidence" in reason


def test_should_refuse_in_scope_and_confident_does_not_refuse():
    with (
        patch("services.agent.guardrails.classify_scope") as mock_classify,
        patch("services.agent.guardrails.get_settings") as mock_settings,
    ):
        mock_classify.return_value = ScopeClassification(in_scope=True, reason="about PSN02")
        mock_settings.return_value.rerank_threshold = 0.0
        refuse, reason = should_refuse("What does PSN02 require?", [make_chunk(0.9)])
        assert refuse is False
