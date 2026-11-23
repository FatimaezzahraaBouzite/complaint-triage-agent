"""Render README.md from report/README.template.md and the results the pipeline saved.

Every number, table and chart in the README comes from these files, written by the step that produces it:

  data/build_summary.json      python -m triage.data        files read, filtering funnel, split
  analysis/eda_summary.json    python -m analysis.eda       analysis tables (+ analysis/relief_by_product.png)
  analysis/baseline.json       python -m triage.baseline    baselines on the full held-out set
  evals/runs/*.json            python -m evals.run_eval     one file per evaluation run, kept in git

Each of those commands rebuilds the README when it finishes. You can also rebuild it at any time:

  python -m report.readme
  python -m report.readme --import-run evals/results_claude.json   # turn an older results.json into a run record

To change the wording, edit report/README.template.md (Jinja2), never README.md: it is overwritten.
"""
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from triage import config

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- formatting helpers used in the template
def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def num(x):
    return "n/a" if x is None else f"{x:,}"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pts(x):
    """Difference between two proportions, in percentage points."""
    return f"{abs(100 * x):.1f}"


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion p measured on n cases."""
    if p is None or not n:
        return None
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"lo": max(0.0, c - h), "hi": min(1.0, c + h)}


def ci_text(ci):
    return "" if ci is None else f"{ci['lo']:.2f} to {ci['hi']:.2f}"


def overlap(a, b):
    return a is not None and b is not None and a["lo"] <= b["hi"] and b["lo"] <= a["hi"]


def file_label(name):
    """CCDB_Export_6_April_2024_through_July_2024.zip -> '6: April 2024 to July 2024'."""
    m = re.match(r"CCDB_Export_(\d+)_(.+?)(\.zip)?$", name.split(" > ")[0], re.I)
    if not m:
        return name
    return f"{m.group(1)}: " + m.group(2).replace("_through_", " to ").replace("_", " ")


def archive_number(name):
    m = re.match(r"CCDB_Export_(\d+)_", name, re.I)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- loading what the pipeline saved
def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def data_context(build):
    if not build:
        return None
    files = sorted(build.get("files_read", []), key=lambda f: (archive_number(f["file"]) or 0, f["file"]))
    for f in files:
        f["label"] = file_label(f["file"])
        f["number"] = archive_number(f["file"])
        f["period"] = f["label"].split(": ", 1)[-1]
        f["share"] = f["narratives"] / f["rows"] if f["rows"] else None
    totals = {k: sum(f[k] for f in files) for k in ("rows", "narratives", "kept", "after_file_sampling")}
    numbers = [archive_number(n) for n in [f["file"] for f in files] + build.get("files_not_opened", [])]
    numbers = sorted({n for n in numbers if n is not None})
    used = [archive_number(f["file"]) for f in files if f["kept"] > 0 and archive_number(f["file"]) is not None]
    last_used = max(used) if used else None  # later files only add complaints outside the date range
    missing = [n for n in range(numbers[0], last_used + 1) if n not in numbers] if numbers and last_used else []
    with_text = [f for f in files if f["kept"] > 0]
    steps = build.get("normalize_steps", [])
    dedup = next((s["count"] for s in steps if "duplicates" in s["label"]), None)
    return {**build, "files": files, "totals": totals, "missing_archive_numbers": missing,
            "share_first": with_text[0] if with_text else None, "share_last": with_text[-1] if with_text else None,
            "after_dedup": dedup, "skipped_count": len(build.get("files_skipped", [])),
            "not_opened_labels": [file_label(n) for n in build.get("files_not_opened", [])]}


def eda_context(eda):
    if not eda:
        return None
    prods, comps, issues = eda["by_product"], eda["top_companies"], eda["top_issues"]
    by = lambda rows, key: sorted(rows, key=lambda r: r[key])
    return {**eda,
            "prod_money_lo": by(prods, "monetary_relief")[0], "prod_money_hi": by(prods, "monetary_relief")[-1],
            "prod_nonmoney_hi": by(prods, "non_monetary_relief")[-1], "prod_timely_lo": by(prods, "timely")[0],
            "comp_money_lo": by(comps, "monetary_relief")[0], "comp_money_hi": by(comps, "monetary_relief")[-1],
            "comp_timely_lo": by(comps, "timely")[0],
            "issue_money_hi": by(issues, "monetary_relief")[-1], "issue_nonmoney_hi": by(issues, "non_monetary_relief")[-1],
            "products_by_money": list(reversed(by(prods, "monetary_relief")))}


def load_runs(folder):
    runs = []
    for p in sorted(Path(folder).glob("*.json")):
        r = read_json(p)
        if r and "summary" in r:
            r["file"] = p.name
            runs.append(r)
    return sorted(runs, key=lambda r: r["created_at"])


def eval_context(runs, fingerprint):
    llm_runs = [r for r in runs if r.get("uses_llm")]
    if not llm_runs:
        return None
    run = llm_runs[-1]
    s, n = run["summary"], run["summary"]["complaints"]
    knn = next((r for r in reversed(runs) if not r.get("uses_llm") and r["per_product"] == run["per_product"]
                and r["seed"] == run["seed"] and r.get("data_fingerprint") == run.get("data_fingerprint")), None)
    a, lr, maj = s["agent"], s["logreg_baseline"], s["majority_baseline"]
    k = knn["summary"]["agent"] if knn else None
    ci = {"agent_product": wilson(a["product_accuracy"], n), "agent_issue": wilson(a["issue_accuracy"], n),
          "lr_product": wilson(lr["product_accuracy"], n),
          "knn_product": wilson(k["product_accuracy"], n) if k else None,
          "knn_issue": wilson(k["issue_accuracy"], n) if k else None}
    return {"run": run, "s": s, "n": n, "a": a, "lr": lr, "maj": maj, "knn": knn, "k": k, "ci": ci,
            "lr_f1": (lr.get("monetary_relief") or {}).get("f1"),
            "stale": fingerprint is not None and run.get("data_fingerprint") not in (None, fingerprint),
            "product_overlap_lr": overlap(ci["agent_product"], ci["lr_product"]),
            "issue_overlap_knn": overlap(ci["agent_issue"], ci["knn_issue"])}


def comparison_chart(ev, path):
    """Product and issue accuracy for each method on the same complaints, with 95% intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [("LLM agent", ev["a"]["product_accuracy"], ev["a"]["issue_accuracy"], "#2a78d6")]
    if ev["k"]:
        methods.append(("Nearest-neighbour\n(no LLM)", ev["k"]["product_accuracy"], ev["k"]["issue_accuracy"], "#7aa6d8"))
    methods += [("TF-IDF + LR", ev["lr"]["product_accuracy"], None, "#9aa3ad"),
                ("Majority class", ev["maj"]["product_accuracy"], None, "#cfd4da")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, idx) in zip(axes, (("Product (8 classes)", 1), ("Issue (79 classes)", 2))):
        rows = [(m[0], m[idx], m[3]) for m in methods if m[idx] is not None]
        for i, (label, v, color) in enumerate(rows):
            ci = wilson(v, ev["n"])
            ax.bar(i, v, color=color, width=0.65)
            ax.errorbar(i, v, yerr=[[v - ci["lo"]], [ci["hi"] - v]], color="#333", capsize=4, lw=1)
            ax.text(i, ci["hi"] + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Accuracy (95% interval)")
    fig.suptitle(f"Accuracy on {ev['n']} held-out complaints ({ev['run']['model']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- render
def build_context(root):
    root = Path(root)
    build = read_json(root / "data" / "build_summary.json")
    runs = load_runs(root / "evals" / "runs")
    ctx = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "cfg": {"products": config.PRODUCTS, "since": config.DATE_MIN, "test_fraction": config.TEST_FRACTION,
                   "model": config.MODEL, "outcomes": config.OUTCOMES},
           "data": data_context(build), "eda": eda_context(read_json(root / "analysis" / "eda_summary.json")),
           "base": read_json(root / "analysis" / "baseline.json"),
           "ev": eval_context(runs, (build or {}).get("fingerprint")), "history": runs}
    if ctx["ev"]:
        chart = root / "evals" / "comparison.png"
        comparison_chart(ctx["ev"], chart)
        ctx["ev"]["chart"] = "evals/comparison.png"
    return ctx


def render(root=config.ROOT, template_dir=HERE):
    env = Environment(loader=FileSystemLoader(str(template_dir)), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters.update(pct=pct, num=num, f3=f3, pts=pts, ci=ci_text)
    text = env.get_template("README.template.md").render(**build_context(root))
    return re.sub(r"\n{3,}", "\n\n", text)


def update(root=config.ROOT):
    path = Path(root) / "README.md"
    path.write_text(render(root), encoding="utf-8")
    print(f"Updated {path}")
    return path


def update_quietly(root=config.ROOT):
    """Called at the end of each pipeline step: a README problem must never break the pipeline."""
    try:
        update(root)
    except Exception as e:  # noqa: BLE001
        print(f"README not updated ({type(e).__name__}: {e}). Run: python -m report.readme")


def import_run(results_path, model=None, root=config.ROOT):
    """Convert a results.json written before run records existed into evals/runs/<...>.json."""
    from evals.run_eval import save_run
    from triage.data import fingerprint
    old = json.loads(Path(results_path).read_text())
    summary, runs = old["summary"], old.get("runs", [])
    if model:
        summary["model"] = model
    products = len({r["true_product"] for r in runs}) or len(config.PRODUCTS)
    created = datetime.fromtimestamp(Path(results_path).stat().st_mtime).isoformat(timespec="seconds")
    return save_run(summary, runs, summary["complaints"] // products, 11, fingerprint(Path(root) / "data" / "complaints.db"),
                    Path(root) / "evals", created_at=created)


def main():
    p = argparse.ArgumentParser(description="Rebuild README.md from the saved pipeline results.")
    p.add_argument("--import-run", metavar="RESULTS_JSON", help="register an older evals/results*.json as a run first")
    p.add_argument("--model", help="model name to record with --import-run, if the file has the wrong one")
    a = p.parse_args()
    if a.import_run:
        import_run(a.import_run, a.model)
    update()


if __name__ == "__main__":
    main()