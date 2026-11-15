# Complaint triage agent

An LLM agent that does the first ten minutes of a complaint analyst's job, on real consumer-finance
complaints from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

**For each incoming complaint the agent:**
1. classifies it (product and issue, regulator's taxonomy),
2. looks up similar past complaints and what the company actually did about them,
3. predicts the likely outcome (monetary relief, non-monetary relief, explanation),
4. sets a priority and gets routed to a team,
5. drafts the first reply to the consumer.

**Everything it reads is real**, and so is the answer key: the most recent 15% of complaints are held out,
and the agent's decisions are scored against the real classification and the company's real response.

```
complaint narrative
      |
      v
 Agent loop (triage/agent.py)  <->  LLM with tool calling
      |
      |-- search_similar_complaints   TF-IDF retrieval over historical complaints + their real outcomes
      |-- get_taxonomy                valid products / issues
      |-- get_outcome_stats           historical relief rates by product / issue / company
      `-- submit_triage               the action. Enforced in code: labels must exist in the taxonomy,
                                      priority cannot go below a risk-based floor, routing is a rule table
      |
      v
 triage_decisions table + logs/traces.jsonl   ->   evals/run_eval.py (vs real labels, vs baselines)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # 12 tests, offline (fixture data, no API key)

python -m triage.data              # build the DB from the monthly CFPB exports in data/raw/
python -m analysis.eda             # -> analysis/eda_report.md + relief_by_product.png
python -m triage.baseline          # TF-IDF + logistic regression on the full held-out set

export ANTHROPIC_API_KEY=sk-ant-...
python -m evals.run_eval           # agent vs baselines on 80 held-out complaints -> evals/results.md
streamlit run app/ui.py            # demo: load a real held-out complaint, compare agent vs what happened
uvicorn app.api:app --reload       # REST: POST /triage {"narrative": "..."}
```

**Data.** The complaints are exported by hand from the
[CFPB complaint search](https://www.consumerfinance.gov/data-research/consumer-complaints/search/) as one file per
month (exports are limited to 100,000 rows), named `complaints_2025_January.csv`, `complaints_2025_Feb.csv`, ...
Copy them into `data/raw/`, or run `python -m triage.data --folder <where they are>`. CSV and Excel files both work.
The build keeps complaints that have a narrative and a closed outcome, restricts to the eight products in
`triage/config.py`, samples 2,000 per product (`--per-product`), and holds out the most recent 15%.

Without `ANTHROPIC_API_KEY` the same pipeline runs on `KNNPolicy`: no LLM, just nearest neighbours driving
the same tools. It is the free smoke test and the baseline the LLM agent has to beat.

## How it is evaluated

| Question | Metric | Compared with |
|---|---|---|
| Does it classify correctly? | product accuracy, issue accuracy | TF-IDF + LR, majority class |
| Can it anticipate the outcome? | 3-class accuracy, monetary-relief precision / recall / F1 | TF-IDF + LR, majority class |
| Is it safe to automate? | % filed without a human, guardrail rejections, cases filed below the priority floor (must be 0) | |
| What does it cost? | LLM steps, tokens, seconds per complaint | |

Split is by time (train on the past, test on the most recent), so retrieval never sees a test complaint.

## Deploy

**Docker (Render, Cloud Run, AWS App Runner):** build the database first; the image copies it in.

```bash
docker build -t complaint-triage .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY complaint-triage
```

**Streamlit Community Cloud:** commit `data/complaints.db` (remove it from `.gitignore`; about 12,000
complaints is a few tens of MB), point the app at `app/ui.py`, add the key under Secrets.

Set a spend limit on the API key before sharing a public link.

## Honest limits

- Scope is eight bank / lender / payments products. Credit-bureau complaints (most of the database) are excluded.
- Only complaints whose consumer agreed to publish the narrative are included, so the sample is not
  representative of all complaints.
- The outcome depends on facts the narrative does not contain (the company's own records). Expect outcome
  prediction to be hard; reporting that honestly is part of the project.
- The draft reply is not scored automatically. Next step: an LLM-judge rubric (acknowledges issue, no
  promise, no invented facts) plus a human spot check.
- In the latest months only complaints that closed quickly have an outcome yet, so the held-out set leans
  toward fast closures.
- Retrieval is TF-IDF. Embeddings are the obvious upgrade once the baseline numbers are recorded.

Data: Consumer Financial Protection Bureau, Consumer Complaint Database (public domain, US government).
