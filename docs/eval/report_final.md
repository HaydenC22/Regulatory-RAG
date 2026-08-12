# Evaluation report — final

| metric | value | baseline | Δ |
|---|---|---|---|
| faithfulness | 0.33 | 0.27 | +0.06 |
| answer_relevancy | 0.26 | 0.12 | +0.14 |
| context_precision | 0.33 | 0.03 | +0.31 |
| citation_validity_pct | 100.00% | 100.00% | +0.00% |
| refusal_precision | 0.45 | 0.42 | +0.03 |
| refusal_recall | 0.93 | 0.87 | +0.07 |
| latency_p50_ms | 3494.41 | 2921.86 | +572.55 |
| latency_p95_ms | 43633.81 | 46367.78 | -2733.97 |

Ran against 40 gold-set questions (see docs/eval/gold_set.jsonl). Judge model: cheap-tier gemini/gemini-flash-lite-latest, temperature=0.