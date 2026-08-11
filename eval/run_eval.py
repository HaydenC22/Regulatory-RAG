"""Runs the eval harness for --config baseline|final, writes
docs/eval/report_{config}.md, and records the run in the eval_runs table
(backing GET /api/v1/eval/latest). Run both configs, then see
docs/eval/report_final.md for the baseline-vs-final comparison."""

import argparse
from pathlib import Path

from eval.ragas_harness import EvalResult, default_gold_set_path, load_gold_set, run_eval
from services.db import get_connection

REPORT_DIR = Path("docs/eval")


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def render_report(config: str, result: EvalResult, baseline: EvalResult | None = None) -> str:
    lines = [
        f"# Evaluation report — {config}",
        "",
        "| metric | value" + (" | baseline | Δ" if baseline else "") + " |",
        "|---|---" + ("|---|---" if baseline else "") + "|",
    ]

    def row(label: str, value, base_value=None, pct=False):
        suffix = "%" if pct else ""
        cell = _fmt(value, suffix)
        if baseline is None:
            lines.append(f"| {label} | {cell} |")
        else:
            base_cell = _fmt(base_value, suffix)
            delta = "n/a" if value is None or base_value is None else f"{value - base_value:+.2f}{suffix}"
            lines.append(f"| {label} | {cell} | {base_cell} | {delta} |")

    row("faithfulness", result.faithfulness, baseline.faithfulness if baseline else None)
    row("answer_relevancy", result.answer_relevancy, baseline.answer_relevancy if baseline else None)
    row(
        "context_precision",
        result.context_precision,
        baseline.context_precision if baseline else None,
    )
    row(
        "citation_validity_pct",
        result.citation_validity_pct,
        baseline.citation_validity_pct if baseline else None,
        pct=True,
    )
    row(
        "refusal_precision",
        result.refusal_precision,
        baseline.refusal_precision if baseline else None,
    )
    row("refusal_recall", result.refusal_recall, baseline.refusal_recall if baseline else None)
    row("latency_p50_ms", result.latency_p50_ms, baseline.latency_p50_ms if baseline else None)
    row("latency_p95_ms", result.latency_p95_ms, baseline.latency_p95_ms if baseline else None)

    lines += [
        "",
        f"Ran against {len(result.per_question)} gold-set questions "
        f"(see docs/eval/gold_set.jsonl). Judge model: cheap-tier Claude, temperature=0.",
    ]
    return "\n".join(lines)


def render_chart(final: EvalResult, baseline: EvalResult, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    baseline_vals = [getattr(baseline, m) or 0 for m in metrics]
    final_vals = [getattr(final, m) or 0 for m in metrics]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="baseline")
    ax.bar([i + width / 2 for i in x], final_vals, width, label="final")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS metrics: baseline vs final retrieval config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)


def record_run(config: str, result: EvalResult) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs (config_name, faithfulness, answer_relevancy, context_precision,
                                    citation_validity_pct, refusal_precision, refusal_recall,
                                    latency_p50_ms, latency_p95_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                config,
                result.faithfulness,
                result.answer_relevancy,
                result.context_precision,
                result.citation_validity_pct,
                result.refusal_precision,
                result.refusal_recall,
                result.latency_p50_ms,
                result.latency_p95_ms,
            ),
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["baseline", "final"], required=True)
    parser.add_argument("--gold-set", default=default_gold_set_path())
    args = parser.parse_args()

    gold_set = load_gold_set(args.gold_set)
    result = run_eval(gold_set, config=args.config)
    record_run(args.config, result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"report_{args.config}.md"
    report_path.write_text(render_report(args.config, result), encoding="utf-8")
    print(f"Wrote {report_path}")

    if args.config == "final":
        baseline_result = run_eval(gold_set, config="baseline")
        (REPORT_DIR / "report_baseline.md").write_text(
            render_report("baseline", baseline_result), encoding="utf-8"
        )
        report_path.write_text(render_report("final", result, baseline_result), encoding="utf-8")
        render_chart(result, baseline_result, REPORT_DIR / "baseline_vs_final.png")
        print("Wrote comparison chart and baseline report.")


if __name__ == "__main__":
    main()
