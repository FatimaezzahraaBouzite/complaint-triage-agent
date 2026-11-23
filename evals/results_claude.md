# Eval on 80 held-out real complaints (claude-haiku-5-5)

| Metric | Agent | TF-IDF + LR | Majority class |
|---|---|---|---|
| Product accuracy | 0.912 | 0.85 | 0.125 |
| Issue accuracy | 0.5 | n/a | n/a |
| Outcome accuracy (3 classes) | 0.838 | n/a | 0.9 |
| Monetary relief F1 | 0.333 | 0.333 | 0 |

Filed without human help: 100%. Guardrails: {'submissions_rejected_then_fixed': 3, 'filed_below_priority_floor': 0}. Cost: {'avg_llm_steps': 3.12, 'avg_tokens': 11792, 'avg_latency_s': 7.34}.
