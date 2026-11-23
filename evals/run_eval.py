"""Evaluate the agent on held-out REAL complaints against what really happened.

For each sampled test complaint the agent files a triage; we compare it with the real labels:
  product / issue          the regulator's classification
  predicted_outcome        what the company actually did (monetary relief, non-monetary relief, explanation)
and with two references on the same complaints: majority class and TF-IDF + logistic regression.

  python -m evals.run_eval                    # 10 complaints per product
  python -m evals.run_eval --per-product 25
"""
import argparse
import json
import shutil
import re
import tempfile
from datetime import datetime
from pathlib import Path

from triage import baseline, config, data
from triage.agent import TriageAgent
from triage.llm import get_llm
from triage.tools import Toolbox, minimum_priority, risk_flags

HERE = Path(__file__).resolve().parent


def acc(pairs):
    pairs = list(pairs)
    return round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None


def main(per_product=10, llm=None, db_path=None, out_dir=HERE, seed=11):
    llm = llm or get_llm()
    train, test = baseline.load("train", db_path), baseline.load("test", db_path)
    sample = test.sample(frac=1, random_state=seed).groupby("product").head(per_product).reset_index(drop=True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # decisions are written to a scratch copy of the database
        db = Path(tmp) / "eval.db"
        shutil.copy(db_path or config.DB_PATH, db)
        agent = TriageAgent(llm=llm, toolbox=Toolbox(db))
        runs = []
        for c in sample.to_dict("records"):
            r = agent.triage(c)
            d = r.decision or {}
            runs.append({"complaint_id": c["complaint_id"], "true_product": c["product"], "true_issue": c["issue"],
                         "true_outcome": c["outcome"], "floor": minimum_priority(risk_flags(c["narrative"], c["tags"])),
                         "pred_product": d.get("product"), "pred_issue": d.get("issue"), "pred_outcome": d.get("predicted_outcome"),
                         "priority": d.get("priority"), "team": d.get("team"), "filed": r.decision is not None,
                         "tools": [t["name"] for t in r.tool_calls], "rejections": r.rejections, "steps": r.steps,
                         "tokens": r.input_tokens + r.output_tokens, "latency_s": r.latency_s,
                         "rationale": d.get("rationale"), "draft_reply": d.get("draft_reply")})

    n = len(runs)
    y_true = [r["true_outcome"] == "monetary_relief" for r in runs]
    prod_model, relief_model = baseline.fit(train)
    lr_prod = prod_model.predict(sample["narrative"])
    summary = {
        "model": getattr(llm, "model", "?"), "complaints": n,
        "filed_rate": round(sum(r["filed"] for r in runs) / n, 3),
        "agent": {"product_accuracy": acc((r["pred_product"], r["true_product"]) for r in runs),
                  "issue_accuracy": acc((r["pred_issue"], r["true_issue"]) for r in runs),
                  "outcome_accuracy": acc((r["pred_outcome"], r["true_outcome"]) for r in runs),
                  "monetary_relief": baseline.relief_scores(y_true, [r["pred_outcome"] == "monetary_relief" for r in runs])},
        "majority_baseline": {"product_accuracy": acc((train["product"].mode()[0], t) for t in sample["product"]),
                              "outcome_accuracy": acc((train["outcome"].mode()[0], t) for t in sample["outcome"])},
        "logreg_baseline": {"product_accuracy": acc(zip(lr_prod, sample["product"])),
                            "monetary_relief": baseline.relief_scores(y_true, relief_model.predict(sample["narrative"]))
                            if relief_model is not None else None},
        "guardrails": {"submissions_rejected_then_fixed": sum(r["rejections"] for r in runs),
                       "filed_below_priority_floor": sum(1 for r in runs if r["filed"] and
                                                         config.PRIORITIES.index(r["priority"]) < config.PRIORITIES.index(r["floor"]))},
        "cost": {"avg_llm_steps": round(sum(r["steps"] for r in runs) / n, 2),
                 "avg_tokens": round(sum(r["tokens"] for r in runs) / n),
                 "avg_latency_s": round(sum(r["latency_s"] for r in runs) / n, 2)},
    }
    print(json.dumps(summary, indent=2))
    wrong = [r for r in runs if r["pred_product"] != r["true_product"]]
    print(f"\n{len(wrong)} product errors (first 5):")
    for r in wrong[:5]:
        print(f"  {r['complaint_id']}: predicted {r['pred_product']!r}, actual {r['true_product']!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=2))
    save_run(summary, runs, per_product, seed, data.fingerprint(db_path), out_dir)
    a, m, l = summary["agent"], summary["majority_baseline"], summary["logreg_baseline"]
    (out_dir / "results.md").write_text("\n".join([
        f"# Eval on {n} held-out real complaints ({summary['model']})", "",
        "| Metric | Agent | TF-IDF + LR | Majority class |", "|---|---|---|---|",
        f"| Product accuracy | {a['product_accuracy']} | {l['product_accuracy']} | {m['product_accuracy']} |",
        f"| Issue accuracy | {a['issue_accuracy']} | n/a | n/a |",
        f"| Outcome accuracy (3 classes) | {a['outcome_accuracy']} | n/a | {m['outcome_accuracy']} |",
        f"| Monetary relief F1 | {a['monetary_relief']['f1']} | {(l['monetary_relief'] or {}).get('f1', 'n/a')} | 0 |",
        "", f"Filed without human help: {summary['filed_rate']:.0%}. Guardrails: {summary['guardrails']}. Cost: {summary['cost']}.", ""]))
    return summary


def save_run(summary, runs, per_product, seed, fingerprint, out_dir=HERE, created_at=None):
    """Keep every run in evals/runs/ (committed to git), so the README and the run history can be rebuilt."""
    created_at = created_at or datetime.now().isoformat(timespec="seconds")
    record = {"created_at": created_at, "model": summary["model"], "uses_llm": not str(summary["model"]).startswith("knn"),
              "per_product": per_product, "seed": seed, "data_fingerprint": fingerprint, "summary": summary, "runs": runs}
    folder = Path(out_dir) / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", str(summary["model"]).lower()).strip("-")
    path = folder / f"{created_at[:19].replace(':', '').replace('-', '')}_{slug}_n{summary['complaints']}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"Saved this run to {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-product", type=int, default=10)
    main(per_product=p.parse_args().per_product)
    from report.readme import update_quietly
    update_quietly()