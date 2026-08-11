"""RAGAS-based evaluation harness. Runs the live pipeline (retrieval + generation
+ citation validation) for every gold-set question under a given retrieval
config, then scores the results with RAGAS (faithfulness, answer_relevancy,
context_precision) plus custom code-computed metrics that RAGAS doesn't cover:
citation validity %, retrieval latency percentiles, and refusal precision/recall
on the out-of-scope bucket."""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from services.agent.qa import answer_question
from services.citation.validator import citation_validity_rate, validate_all
from services.llm import get_chat_model
from services.retrieval.pipeline import RetrievalConfig


@dataclass
class GoldRecord:
    id: str
    question: str
    question_type: str
    gold_answer: str
    gold_citations: list[dict]
    is_answerable: bool


@dataclass
class EvalResult:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    citation_validity_pct: float
    refusal_precision: float
    refusal_recall: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_question: list[dict] = field(default_factory=list)


def load_gold_set(path: str) -> list[GoldRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(GoldRecord(**data))
    return records


def _judge_model() -> BaseChatModel:
    """Uses the same cheap-tier chat model as generation (provider-agnostic —
    see services/llm.py and ADR-0004), documented as a methodology note since
    judge choice affects RAGAS scores."""
    return get_chat_model(tier="cheap")


def run_eval(gold_set: list[GoldRecord], config: RetrievalConfig) -> EvalResult:
    ragas_rows = []
    per_question = []
    latencies = []
    citation_rates = []

    refusal_tp = refusal_fp = refusal_tn = refusal_fn = 0

    for record in gold_set:
        result = answer_question(record.question, config=config)
        latencies.append(result.latency_ms)

        checks = validate_all(result.citations, result.retrieved_chunks)
        rate = citation_validity_rate(checks) if result.citations else (0.0 if not result.refused else 1.0)
        citation_rates.append(rate)

        expected_refusal = not record.is_answerable
        actual_refusal = result.refused
        if expected_refusal and actual_refusal:
            refusal_tp += 1
        elif expected_refusal and not actual_refusal:
            refusal_fn += 1
        elif not expected_refusal and actual_refusal:
            refusal_fp += 1
        else:
            refusal_tn += 1

        if record.is_answerable:
            ragas_rows.append(
                {
                    "question": record.question,
                    "answer": result.answer,
                    "contexts": [c.text for c in result.retrieved_chunks] or [""],
                    "ground_truth": record.gold_answer,
                }
            )

        per_question.append(
            {
                "id": record.id,
                "question_type": record.question_type,
                "refused": actual_refusal,
                "expected_refusal": expected_refusal,
                "citation_validity": rate,
                "latency_ms": result.latency_ms,
            }
        )

    ragas_scores: dict[str, float] = {}
    if ragas_rows:
        dataset = Dataset.from_list(ragas_rows)
        judge = _judge_model()
        report = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge,
        )
        ragas_scores = {k: float(v) for k, v in report.to_pandas().mean(numeric_only=True).to_dict().items()}

    precision_denom = refusal_tp + refusal_fp
    recall_denom = refusal_tp + refusal_fn
    refusal_precision = refusal_tp / precision_denom if precision_denom else 0.0
    refusal_recall = refusal_tp / recall_denom if recall_denom else 0.0

    sorted_latencies = sorted(latencies)
    p50 = statistics.median(sorted_latencies) if sorted_latencies else 0.0
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0

    return EvalResult(
        faithfulness=ragas_scores.get("faithfulness"),
        answer_relevancy=ragas_scores.get("answer_relevancy"),
        context_precision=ragas_scores.get("context_precision"),
        citation_validity_pct=statistics.mean(citation_rates) * 100 if citation_rates else 0.0,
        refusal_precision=refusal_precision,
        refusal_recall=refusal_recall,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        per_question=per_question,
    )


def default_gold_set_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "docs" / "eval" / "gold_set.jsonl")
