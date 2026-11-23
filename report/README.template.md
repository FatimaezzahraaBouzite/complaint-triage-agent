<!-- GENERATED FILE. Do not edit README.md by hand: edit report/README.template.md and run `python -m report.readme`.
     Numbers, tables and charts are filled in from data/build_summary.json, analysis/eda_summary.json,
     analysis/baseline.json and evals/runs/*.json. Last rendered {{ generated_at }}. -->

# Complaint triage agent

An LLM agent that does the first ten minutes of a complaint analyst's job on **real consumer-finance
complaints** from the US Consumer Financial Protection Bureau (CFPB). It reads a complaint, classifies it,
looks up similar past cases and what companies actually did about them, predicts the likely outcome, sets a
priority, routes it to a team and drafts the first reply. Every decision is scored against what really
happened to held-out complaints, and compared with classic machine-learning baselines.

{% if ev %}
{% set a, lr, maj, k, n = ev.a, ev.lr, ev.maj, ev.k, ev.n %}
**Results at a glance** ({{ n }} held-out complaints, {{ ev.run.per_product }} per product, model `{{ ev.run.model }}`):

| Metric | LLM agent | Nearest-neighbour policy (no LLM) | TF-IDF + logistic regression | Majority class |
|---|---|---|---|---|
| Product accuracy ({{ cfg.products | length }} classes) | {{ a.product_accuracy | f3 }} | {{ k.product_accuracy | f3 if k else "not run" }} | {{ lr.product_accuracy | f3 }} | {{ maj.product_accuracy | f3 }} |
| Issue accuracy{% if data %} ({{ data.issues }} classes){% endif %} | {{ a.issue_accuracy | f3 }} | {{ k.issue_accuracy | f3 if k else "not run" }} | not trained | n/a |
| Outcome accuracy (3 classes) | {{ a.outcome_accuracy | f3 }} | {{ k.outcome_accuracy | f3 if k else "not run" }} | not trained | {{ maj.outcome_accuracy | f3 }} |
| Monetary relief F1 | {{ a.monetary_relief.f1 | f3 }} | {{ k.monetary_relief.f1 | f3 if k else "not run" }} | {{ ev.lr_f1 | f3 }} | 0.000 |
| Filed without a human | {{ ev.s.filed_rate | pct(0) }} | {{ ev.knn.summary.filed_rate | pct(0) if k else "not run" }} | n/a | n/a |
| Filed below the risk-based priority floor | {{ ev.s.guardrails.filed_below_priority_floor }} | {{ ev.knn.summary.guardrails.filed_below_priority_floor if k else "not run" }} | n/a | n/a |
| Cost per complaint | {{ ev.s.cost.avg_llm_steps }} LLM calls, {{ ev.s.cost.avg_tokens | num }} tokens, {{ ev.s.cost.avg_latency_s }} s | {{ (ev.knn.summary.cost.avg_latency_s ~ " s") if k else "not run" }} | under 0.1 s | n/a |

{% set best_product = a.product_accuracy >= lr.product_accuracy and (not k or a.product_accuracy >= k.product_accuracy) %}
{% set best_issue = not k or a.issue_accuracy >= k.issue_accuracy %}
In short: {% if best_product and best_issue %}the agent is the best classifier tested on products and issues{% elif best_issue %}the agent is the best method on issues, but not on products{% elif best_product %}the agent is the best method on products, but not on issues{% else %}the agent does not beat the simpler methods on classification yet{% endif %}, and {% if a.outcome_accuracy > maj.outcome_accuracy %}it also predicts outcomes better than "always say explanation"{% else %}no method, including the agent, predicts outcomes better than "always say explanation"{% endif %}. Details, caveats and confidence intervals are in [Results](#5-results).
{% if ev.stale %}

> **Note:** the data was rebuilt after this evaluation run and the held-out set changed. Re-run `python -m evals.run_eval` to refresh these numbers.
{% endif %}
{% else %}
**Results:** not run yet. Run `python -m evals.run_eval` (see [Reproduce it](#8-reproduce-it)).
{% endif %}

---

## Contents
1. [The problem](#1-the-problem)
2. [Data: source, preparation and cleaning](#2-data-source-preparation-and-cleaning)
3. [Exploratory analysis](#3-exploratory-analysis)
4. [The agent and the baselines](#4-the-agent-and-the-baselines)
5. [Results](#5-results)
6. [Limitations](#6-limitations)
7. [Next steps](#7-next-steps)
8. [Reproduce it](#8-reproduce-it)
9. [Project structure](#9-project-structure)
10. [Deploy](#10-deploy)

---

## 1. The problem

Banks, lenders and payment companies receive complaints as free text. Before anyone can fix the problem,
an analyst has to read the complaint, file it under the right product and issue, judge how urgent it is,
send it to the right team, and reply to the consumer.{% if eda %} In this data the median complaint is
{{ eda.median_words }} words, so reading alone takes about {{ eda.reading_minutes_per_complaint }} minutes; the research and writing take longer.{% endif +%}

The agent automates that first pass. For each complaint it must:

1. **Classify** it by product and issue, using the regulator's own taxonomy{% if data %} ({{ data.products }} products, {{ data.issues }} issues){% endif %}.
2. **Investigate**: retrieve the most similar past complaints and their real outcomes, and historical
   relief rates for the product, issue and company.
3. **Predict the outcome**: monetary relief, non-monetary relief, or closed with an explanation only.
4. **Prioritise and route**: set low, medium or high priority, with a floor enforced in code for risk
   signals (fraud, legal threats, foreclosure or repossession, vulnerable consumers).
5. **Draft the first reply** to the consumer.

---

## 2. Data: source, preparation and cleaning

### Source

The [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
is public. Consumers can opt in to having their written account, the *narrative*, published.

On **14 August 2026 the CFPB stopped publishing narratives** in the live database, its API and its
exports, so current exports from the website have no narrative column and cannot be used. The narratives
published before that date are kept in the CFPB's
[Consumer Complaint Database Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/),
as one zip file per period. That archive is the data source for this project.

{% if data %}
### Files and filtering funnel

The build (`python -m triage.data`) reads the zips directly, without unzipping, and filters while reading,
so {{ (data.totals.rows / 1e6) | round(1) }} million rows fit in a laptop's memory.
{% if data.files_not_opened %}
{{ data.files_not_opened | length }} archive file(s) ending before {{ data.since }} were not opened ({{ data.not_opened_labels | join("; ") }}).
{% endif %}
{% if data.skipped_count %}
{{ data.skipped_count }} other file(s) in the folder had no narrative column and were skipped.
{% endif %}

| Archive file | Rows | With a narrative | Kept (in scope) |
|---|---:|---:|---:|
{% for f in data.files %}
| {{ f.label }} | {{ f.rows | num }} | {{ f.narratives | num }} ({{ f.share | pct }}) | {{ f.kept | num }} |
{% endfor %}
| **Total** | **{{ data.totals.rows | num }}** | **{{ data.totals.narratives | num }}** | **{{ data.totals.kept | num }}** |

| Step | Complaints left |
|---|---:|
| All rows in the archive files read | {{ data.totals.rows | num }} |
| Has a consumer narrative | {{ data.totals.narratives | num }} |
| In scope: narrative of 50+ characters, closed with a known outcome, one of the {{ cfg.products | length }} products, received on or after {{ data.since }} | {{ data.totals.kept | num }} |
| Sampled to at most {{ data.per_file_cap | num }} per product in each file (keeps memory bounded) | {{ data.totals.after_file_sampling | num }} |
| Duplicates removed, then sampled to {{ data.per_product_cap | num }} per product | **{{ data.complaints | num }}** |

### Cleaning and design decisions

| Decision | Why |
|---|---|
| **Scope: {{ cfg.products | length }} bank, lender and payments products** ({{ cfg.products | join("; ") }}) | This is the complaint desk of a consumer-finance company. Credit-bureau complaints, most of the database, are left out: many are templated dispute letters and they belong to a different business. |
| **Start date {{ data.since }}** | The CFPB renamed several products in 2023 (for example "Credit card or prepaid card" became "Credit card"). One consistent label set matters more than extra history. |
| **Narrative of at least 50 characters** | The narrative is the agent's only input. |
| **Closed outcomes only** ({{ cfg.outcomes.keys() | join(", ") }}) | Complaints still in progress or untimely have no outcome to score against. These map to the three classes `explanation`, `monetary_relief` and `non_monetary_relief`. |
| **{{ data.per_product_cap | num }} complaints per product** | Equal classes, and a data set that runs on a laptop. |
| **Time-based split**: the most recent {{ (100 * data.test_fraction) | round | int }}% held out | Train on the past, test on the future, as in production. Retrieval only ever searches the training part. Train: {{ data.split.train.complaints | num }} complaints, {{ data.split.train.date_min }} to {{ data.split.train.date_max }}. Test: {{ data.split.test.complaints | num }} complaints, {{ data.split.test.date_min }} to {{ data.split.test.date_max }}. |
| Dates parsed from ISO, US and Excel formats; text fields stripped; empty values normalised | The archive and the website use different formats. |

### Observations about the data itself

{% if data.share_first and data.share_last %}
- **Share of rows with a published narrative:** {{ data.share_first.share | pct }} in {{ data.share_first.period }} (archive file {{ data.share_first.number }}), against {{ data.share_last.share | pct }} in {{ data.share_last.period }} (file {{ data.share_last.number }}).
{% endif %}
{% if data.missing_archive_numbers %}
- **Archive files not in this build:** number(s) {{ data.missing_archive_numbers | join(", ") }} were not in the folder, so those periods are missing.
{% else %}
- **No gaps:** every archive file between the first and the last one used is present.
{% endif %}
{% else %}
_Data summary not available yet: run `python -m triage.data` (see [Reproduce it](#8-reproduce-it))._
{% endif %}

---

## 3. Exploratory analysis

{% if eda %}
Generated by `python -m analysis.eda` (full output in [`analysis/eda_report.md`](analysis/eda_report.md)).
All figures describe the {{ eda.complaints | num }}-complaint sample.

**Overview**

- **{{ eda.complaints | num }} complaints**, received {{ eda.date_min }} to {{ eda.date_max }}, **{{ eda.companies | num }} companies**, {{ eda.products }} products, {{ eda.issues }} issues.
- **Outcome mix:** explanation {{ eda.outcome_mix.get("explanation", 0) | pct }}, non-monetary relief {{ eda.outcome_mix.get("non_monetary_relief", 0) | pct }}, monetary relief {{ eda.outcome_mix.get("monetary_relief", 0) | pct }}.
- **Narrative length:** median {{ eda.median_words }} words, 90th percentile {{ eda.p90_words }} words.
- **Reading time alone:** about {{ eda.reading_minutes_per_complaint }} minutes per complaint at {{ eda.reading_wpm }} words per minute, or {{ eda.reading_hours | round | int | num }} analyst-hours for this sample, before any research or writing.
- **Risk signals:** {% for k_, v in eda.risk_signals | dictsort(by="value", reverse=true) %}{{ k_ | replace("_", " ") }} {{ v | pct }}{{ ", " if not loop.last }}{% endfor %}.
- **Priority floor from those signals:** {% for level in ["low", "medium", "high"] %}{{ level }} {{ eda.priority_floor.get(level, 0) | pct }}{{ ", " if not loop.last }}{% endfor %}.

**Relief by product**

![How often does the consumer get relief?](analysis/{{ eda.figure }})

| Product | Complaints | Median words | Monetary relief | Non-monetary relief | Timely response |
|---|---:|---:|---:|---:|---:|
{% for r in eda.products_by_money %}
| {{ r.name }} | {{ r.complaints | int | num }} | {{ r.median_words | int }} | {{ r.monetary_relief | pct }} | {{ r.non_monetary_relief | pct }} | {{ r.timely | pct }} |
{% endfor %}

**Most frequent issues**

| Issue | Complaints | Monetary relief | Non-monetary relief | Timely |
|---|---:|---:|---:|---:|
{% for r in eda.top_issues %}
| {{ r.name }} | {{ r.complaints | int | num }} | {{ r.monetary_relief | pct }} | {{ r.non_monetary_relief | pct }} | {{ r.timely | pct }} |
{% endfor %}

**Largest companies in the sample**

| Company | Complaints | Monetary relief | Non-monetary relief | Timely |
|---|---:|---:|---:|---:|
{% for r in eda.top_companies %}
| {{ r.name }} | {{ r.complaints | int | num }} | {{ r.monetary_relief | pct }} | {{ r.non_monetary_relief | pct }} | {{ r.timely | pct }} |
{% endfor %}

**What the analysis says**

- **Relief depends heavily on product, issue and company.** Monetary relief ranges from {{ eda.prod_money_lo.monetary_relief | pct }} ({{ eda.prod_money_lo.name }}) to {{ eda.prod_money_hi.monetary_relief | pct }} ({{ eda.prod_money_hi.name }}) by product, and from {{ eda.comp_money_lo.monetary_relief | pct }} ({{ eda.comp_money_lo.name }}) to {{ eda.comp_money_hi.monetary_relief | pct }} ({{ eda.comp_money_hi.name }}) among the largest companies. Among the most frequent issues, "{{ eda.issue_money_hi.name }}" is the most likely route to money back ({{ eda.issue_money_hi.monetary_relief | pct }}).
- **Non-monetary relief follows a different pattern:** it is highest for {{ eda.prod_nonmoney_hi.name }} ({{ eda.prod_nonmoney_hi.non_monetary_relief | pct }}), and for the issue "{{ eda.issue_nonmoney_hi.name }}" ({{ eda.issue_nonmoney_hi.non_monetary_relief | pct }}).
- **Response times vary little, except in one corner:** {{ eda.prod_timely_lo.name }} has the lowest timely-response rate ({{ eda.prod_timely_lo.timely | pct }}); among the largest companies the lowest is {{ eda.comp_timely_lo.name }} ({{ eda.comp_timely_lo.timely | pct }}).
- **Implication for the agent:** the narrative tells you *what* the complaint is about much better than
  *how it will end*. That is why the agent is given outcome statistics by product, issue and company as a
  tool, rather than asked to guess outcomes from text.
{% else %}
_Analysis not available yet: run `python -m analysis.eda`._
{% endif %}

---

## 4. The agent and the baselines

### How the agent works

```
complaint narrative
      |
      v
 Agent loop (triage/agent.py)  <->  Claude with tool calling ({{ ev.run.model if ev else cfg.model }})
      |
      |-- search_similar_complaints   TF-IDF retrieval over the training complaints, with their real outcomes
      |-- get_taxonomy                valid products and issues
      |-- get_outcome_stats           historical relief rates by product / issue / company (min. 20 cases)
      `-- submit_triage               the action
      |
      v
 triage_decisions table + logs/traces.jsonl   ->   evals/run_eval.py
```

The model chooses which tools to call and what to conclude. **Business rules live in code, not in the
prompt**, so the model cannot talk its way past them:

- `submit_triage` rejects any product or issue that is not in the taxonomy, and returns the valid values
  so the model can correct itself.
- A **priority floor** is computed from risk signals in the text and consumer tags: fraud or identity
  theft, legal threats, foreclosure, repossession or garnishment, older Americans and servicemembers.
  A submission below the floor is rejected.
- **Routing to a team** is a deterministic rule table, not a model choice.
- The loop has a step budget. If the model stops without filing, it gets one reminder, then the case is
  handed to a human.
- Complaint text is treated as data: the prompt tells the model to ignore any instructions inside it.

### Baselines

All baselines are trained on the same training complaints the agent can search.

| Baseline | What it is | Why it is here |
|---|---|---|
| **Majority class** | Always predicts the most common label | The floor any method must clear; for outcomes it is a hard bar because most complaints close with an explanation |
| **TF-IDF + logistic regression** (`triage/baseline.py`) | Product classifier, plus a class-balanced "will this end in monetary relief?" classifier | The standard cheap text model. If the agent cannot beat it, the LLM is not worth its cost for that task |
| **Nearest-neighbour policy** (`KNNPolicy` in `triage/llm.py`) | No LLM: drives the *same tools* with fixed rules (search, majority vote of the 5 nearest cases, historical base rate, submit) | Isolates what the LLM adds over the tools themselves. Also lets the whole pipeline and the tests run with no API key |

### Evaluation design

- **Held-out data only:** the evaluation samples from the most recent complaints, which retrieval never sees.
- **Same complaints for every method:** a fixed number per product, drawn with a fixed seed, so every
  column of the results table scores the same cases.
- **Scored against reality:** the CFPB's product and issue labels, and the company's actual response.
- **Safety metrics as well as accuracy:** cases filed without a human, guardrail rejections, and cases
  filed below the priority floor (must be zero).
- **Cost:** model calls, tokens and seconds per complaint.
- **Every run is kept** in `evals/runs/`, so results can be compared over time (see the run history below).

---

## 5. Results

{% if ev %}
{% set a, lr, maj, k, n, ci = ev.a, ev.lr, ev.maj, ev.k, ev.n, ev.ci %}
### On the {{ n }}-complaint evaluation sample

Run `{{ ev.run.file }}`, {{ ev.run.created_at[:10] }}{% if ev.knn %}; no-LLM run `{{ ev.knn.file }}`{% endif %}.

![Accuracy by method]({{ ev.chart }})

| Metric | LLM agent | Nearest-neighbour policy | TF-IDF + LR | Majority class |
|---|---|---|---|---|
| Product accuracy | {{ a.product_accuracy | f3 }} ({{ ci.agent_product | ci }}) | {{ (k.product_accuracy | f3) ~ " (" ~ (ci.knn_product | ci) ~ ")" if k else "not run" }} | {{ lr.product_accuracy | f3 }} ({{ ci.lr_product | ci }}) | {{ maj.product_accuracy | f3 }} |
| Issue accuracy | {{ a.issue_accuracy | f3 }} ({{ ci.agent_issue | ci }}) | {{ (k.issue_accuracy | f3) ~ " (" ~ (ci.knn_issue | ci) ~ ")" if k else "not run" }} | not trained | n/a |
| Outcome accuracy | {{ a.outcome_accuracy | f3 }} | {{ k.outcome_accuracy | f3 if k else "not run" }} | not trained | {{ maj.outcome_accuracy | f3 }} |
| Monetary relief precision / recall / F1 | {{ a.monetary_relief.precision | f3 }} / {{ a.monetary_relief.recall | f3 }} / {{ a.monetary_relief.f1 | f3 }} | {{ (k.monetary_relief.precision | f3) ~ " / " ~ (k.monetary_relief.recall | f3) ~ " / " ~ (k.monetary_relief.f1 | f3) if k else "not run" }} | {% if lr.monetary_relief %}{{ lr.monetary_relief.precision | f3 }} / {{ lr.monetary_relief.recall | f3 }} / {{ lr.monetary_relief.f1 | f3 }}{% else %}n/a{% endif %} | 0 |
| Submissions rejected by a guardrail, then fixed | {{ ev.s.guardrails.submissions_rejected_then_fixed }} | {{ ev.knn.summary.guardrails.submissions_rejected_then_fixed if k else "not run" }} | n/a | n/a |
| Filed below the priority floor | {{ ev.s.guardrails.filed_below_priority_floor }} | {{ ev.knn.summary.guardrails.filed_below_priority_floor if k else "not run" }} | n/a | n/a |
| LLM calls / tokens / seconds per complaint | {{ ev.s.cost.avg_llm_steps }} / {{ ev.s.cost.avg_tokens | num }} / {{ ev.s.cost.avg_latency_s }} | {{ (ev.knn.summary.cost.avg_llm_steps ~ " / 0 / " ~ ev.knn.summary.cost.avg_latency_s) if k else "not run" }} | n/a | n/a |

Values in brackets are 95% Wilson intervals for a proportion out of {{ n }}.

{% if base %}
### On the full held-out test set ({{ base.test | num }} complaints, baselines only)

| Metric | TF-IDF + LR | Majority class |
|---|---|---|
| Product accuracy | {{ base.product_accuracy | f3 }} | {{ base.majority_product_accuracy | f3 }} |
| Monetary relief rate in the test set | {{ base.monetary_relief_rate | f3 }} | |
{% if base.monetary_relief %}
| Monetary relief, balanced classifier | precision {{ base.monetary_relief.precision | f3 }}, recall {{ base.monetary_relief.recall | f3 }}, F1 {{ base.monetary_relief.f1 | f3 }} | |
{% endif %}
{% endif %}

### What the results mean

**Classification.**
On product the agent scores {{ a.product_accuracy | pct }}, against {{ lr.product_accuracy | pct }} for logistic regression{% if k %} and {{ k.product_accuracy | pct }} for the nearest-neighbour policy{% endif %} on the same complaints{% if k %}; on issue {{ a.issue_accuracy | pct }} against {{ k.issue_accuracy | pct }}{% endif %}.
{% if ev.product_overlap_lr %}
With {{ n }} complaints the 95% intervals of the agent and logistic regression overlap, so the {{ "lead" if a.product_accuracy >= lr.product_accuracy else "gap" }} on product is not yet statistically settled{% if base %}; for reference, logistic regression scores {{ base.product_accuracy | pct }} on the full test set{% endif %}. A larger run (`--per-product 25`) narrows the intervals.
{% else %}
The 95% intervals of the agent and logistic regression on product do not overlap, so the difference is unlikely to be chance.
{% endif %}
{% if data %}Issue is the harder and more useful task: {{ data.issues }} classes, many of them close.{% endif +%}
{% if k and ev.issue_overlap_knn %}
On issue, the intervals of the agent and the nearest-neighbour policy also overlap, so that {{ "lead" if a.issue_accuracy >= k.issue_accuracy else "gap" }} needs a larger run to confirm.
{% endif %}

{% if k %}
**What the LLM adds beyond the tools.** The nearest-neighbour policy uses exactly the same retrieval and
statistics. The difference between it and the agent ({{ "+" if a.product_accuracy >= k.product_accuracy else "-" }}{{ (a.product_accuracy - k.product_accuracy) | pts }} points on product, {{ "+" if a.issue_accuracy >= k.issue_accuracy else "-" }}{{ (a.issue_accuracy - k.issue_accuracy) | pts }} on issue) is what reading and reasoning add over a vote of similar cases.

{% endif %}
**Outcome prediction.** With {{ maj.outcome_accuracy | pct }} of the evaluation complaints closed with an
explanation, always guessing "explanation" scores {{ maj.outcome_accuracy | pct }}. The agent scores {{ a.outcome_accuracy | pct }}{% if a.outcome_accuracy > maj.outcome_accuracy %}, better than that guess{% else %}, {{ "the same as" if a.outcome_accuracy == maj.outcome_accuracy else "below" }} that guess{% endif %}. On monetary relief, the agent's F1 is {{ a.monetary_relief.f1 | f3 }} against {{ ev.lr_f1 | f3 }} for the balanced logistic regression.
{% if (a.monetary_relief.f1 or 0) < 0.5 and (ev.lr_f1 or 0) < 0.5 %}
Both are weak. The outcome depends on company practice and on facts in the company's records, not just on
what the consumer wrote (see section 3).
{% endif %}

**Guardrails.** {% if ev.s.guardrails.filed_below_priority_floor == 0 %}No complaint was filed below its risk-based priority floor{% if k %} by either method{% endif %}.{% else %}{{ ev.s.guardrails.filed_below_priority_floor }} complaint(s) were filed below their priority floor: this should never happen and needs investigating.{% endif +%}
The agent needed {{ ev.s.guardrails.submissions_rejected_then_fixed }} correction(s) in {{ n }} cases{% if k %}; the rule-based policy needed {{ ev.knn.summary.guardrails.submissions_rejected_then_fixed }} because it always starts at "low" and lets the guardrail raise it{% endif %}.

**Cost and speed.** About {{ ev.s.cost.avg_llm_steps }} model calls, {{ ev.s.cost.avg_tokens | num }} tokens and {{ ev.s.cost.avg_latency_s }} seconds per complaint{% if eda %}, against roughly {{ eda.reading_minutes_per_complaint }} minutes for a person just to read it{% endif %}.

### Run history

| Date | Model | Complaints | Product | Issue | Outcome | Relief F1 | Tokens per complaint | Run file |
|---|---|---:|---:|---:|---:|---:|---:|---|
{% for r in history | reverse %}
| {{ r.created_at[:16] | replace("T", " ") }} | {{ r.model }} | {{ r.summary.complaints }} | {{ r.summary.agent.product_accuracy | f3 }} | {{ r.summary.agent.issue_accuracy | f3 }} | {{ r.summary.agent.outcome_accuracy | f3 }} | {{ r.summary.agent.monetary_relief.f1 | f3 }} | {{ r.summary.cost.avg_tokens | num }} | `{{ r.file }}` |
{% endfor %}
{% else %}
_No evaluation run yet: run `python -m evals.run_eval`._
{% endif %}

---

## 6. Limitations

{% if ev %}
- **Evaluation sample size.** {{ ev.n }} complaints gives intervals of roughly plus or minus {{ (100 * (ev.ci.agent_product.hi - ev.ci.agent_product.lo) / 2) | round | int }} points on accuracy.
{% endif %}
- **No issue-level baseline yet.** Issue accuracy is only compared with the nearest-neighbour policy.
- **Outcome prediction is weak for every method.** The agent should be read as a classifier and router,
  not as a predictor of how much a company will pay.
- **The draft reply is not scored automatically.** It is checked by reading examples in the demo.
- **The sample is not representative of all complaints.** Only consumers who agreed to publish their
  narrative are included, products are balanced by design, and credit-bureau complaints are excluded.
- **A fixed historical snapshot.** The CFPB no longer publishes narratives, so the data cannot be refreshed
  from the public site{% if data %}; it ends on {{ data.date_max }}{% endif %}.{% if data and data.missing_archive_numbers %} Archive file(s) {{ data.missing_archive_numbers | join(", ") }} are missing from this build.{% endif +%}
- **Retrieval is TF-IDF.** Simple and transparent, but weaker than embeddings on paraphrases.

---

## 7. Next steps

1. Run the evaluation on more complaints (`--per-product 25`) to narrow the intervals.
2. Add a TF-IDF + logistic regression **issue** classifier as a baseline.
3. Review the agent's errors and fix the most frequent cause (prompt or retrieval), measuring before and after.
4. Score draft replies with a rubric: names the specific problem, makes no promise, invents no facts.
5. Give the outcome model the company and issue as features, since section 3 shows they carry most of the signal.
{% if data and data.missing_archive_numbers %}
6. Add the missing archive file(s) ({{ data.missing_archive_numbers | join(", ") }}) and rebuild.
{% endif %}

---

## 8. Reproduce it

Commands are for Windows PowerShell; Mac and Linux equivalents are in comments. Each step rewrites this
README with its new numbers when it finishes.

```powershell
python -m venv .venv
.venv\Scripts\activate                     # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # offline tests, no data or API key needed

# Data: download the "CCDB Export" zips from the CFPB narratives archive (link in section 2). Leave them zipped.
python -m triage.data --folder C:\path\to\Downloads    # -> data/complaints.db, data/build_summary.json
python -m analysis.eda                     # -> analysis/eda_summary.json, eda_report.md, relief_by_product.png
python -m triage.baseline                  # -> analysis/baseline.json, baseline_report.md

# Agent evaluation (needs an Anthropic API key; set a spend limit on it first)
$env:ANTHROPIC_API_KEY="sk-ant-..."        # Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...
python -m evals.run_eval                   # -> evals/runs/<date>_<model>_n80.json

# Same complaints, no LLM
$env:AGENT_LLM="offline"; python -m evals.run_eval; Remove-Item Env:AGENT_LLM

# Rebuild this README by hand (for example after editing report/README.template.md)
python -m report.readme

# Demo and API
streamlit run app/ui.py
uvicorn app.api:app --reload               # POST /triage {"narrative": "..."}
```

Useful options: `python -m triage.data --since 2022-01-01` (more history),
`--per-product 5000` (bigger sample), `python -m evals.run_eval --per-product 25` (more evaluation cases).
Set `AGENT_MODEL` to try another Claude model; each run is added to the run history.

---

## 9. Project structure

```
triage/
  config.py      scope (products), outcome classes, split, model name
  data.py        reads the CFPB archive zips, filters, samples, splits, builds data/complaints.db
  tools.py       the agent's tools and the guardrails (taxonomy check, priority floor, routing)
  agent.py       system prompt and the agent loop
  llm.py         Claude backend, and the no-LLM nearest-neighbour policy
  baseline.py    TF-IDF + logistic regression baselines
analysis/
  eda.py         exploratory analysis
evals/
  run_eval.py    held-out evaluation of the agent against the baselines
  runs/          one JSON file per evaluation run (kept in git)
report/
  readme.py            renders README.md from the saved results
  README.template.md   the README's wording; edit this, not README.md
app/
  ui.py          Streamlit demo: triage a real held-out complaint and compare with what happened
  api.py         FastAPI endpoint
tests/           offline tests (fixture data in the archive's format)
```

---

## 10. Deploy

**Streamlit Community Cloud:** commit `data/complaints.db` (remove `data/*.db` from `.gitignore`), create
an app pointing at `app/ui.py`, and add `ANTHROPIC_API_KEY` under Secrets.

**Docker** (Render, Cloud Run, AWS App Runner), after building the database:

```bash
docker build -t complaint-triage .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY complaint-triage
```

Set a monthly spend limit on the API key before sharing a public link.

---

**Data:** Consumer Financial Protection Bureau, Consumer Complaint Database and Narratives Archive
(US government work, public domain). Complaint narratives are the consumers' own unverified accounts,
published by the CFPB with personal information removed.