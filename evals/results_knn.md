# Eval on 80 held-out real complaints (knn-policy (no LLM))

| Metric | Agent | TF-IDF + LR | Majority class |
|---|---|---|---|
| Product accuracy | 0.787 | 0.85 | 0.125 |
| Issue accuracy | 0.438 | n/a | n/a |
| Outcome accuracy (3 classes) | 0.9 | n/a | 0.9 |
| Monetary relief F1 | 0.0 | 0.333 | 0 |

Filed without human help: 100%. Guardrails: {'submissions_rejected_then_fixed': 45, 'filed_below_priority_floor': 0}. Cost: {'avg_llm_steps': 3.56, 'avg_tokens': 0, 'avg_latency_s': 0.26}.
