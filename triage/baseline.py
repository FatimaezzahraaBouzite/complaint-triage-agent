"""Classic-ML baselines trained on the same history the agent can search:
TF-IDF + logistic regression for (1) product and (2) 'will this end in monetary relief?'.

  python -m triage.baseline     # scores on the full held-out test set -> analysis/baseline_report.md
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from . import config


def load(split, db_path=None):
    con = sqlite3.connect(db_path or config.DB_PATH)
    df = pd.read_sql("SELECT * FROM complaints WHERE split = ?", con, params=(split,))
    con.close()
    return df


def fit(train):
    def model(**kw):
        return make_pipeline(TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=80000,
                                             min_df=2 if len(train) > 2000 else 1), LogisticRegression(max_iter=2000, **kw))
    product = model(C=5).fit(train["narrative"], train["product"])
    relief = None
    y = (train["outcome"] == "monetary_relief").astype(int)
    if y.nunique() == 2:
        relief = model(class_weight="balanced").fit(train["narrative"], y)
    return product, relief


def relief_scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3)}


def main(db_path=None, out=config.ROOT / "analysis" / "baseline_report.md"):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    train, test = load("train", db_path), load("test", db_path)
    product, relief = fit(train)
    pred = product.predict(test["narrative"])
    y = (test["outcome"] == "monetary_relief").astype(int)
    prod_acc = accuracy_score(test["product"], pred)
    majority = (test["product"] == train["product"].mode()[0]).mean()
    result = {"train": len(train), "test": len(test), "product_accuracy": round(float(prod_acc), 3),
              "majority_product_accuracy": round(float(majority), 3), "monetary_relief_rate": round(float(y.mean()), 3),
              "monetary_relief": relief_scores(y, relief.predict(test["narrative"])) if relief is not None else None}
    out.with_name("baseline.json").write_text(json.dumps(result, indent=2))
    lines = ["# Baselines on the held-out test set", "",
             f"- Train: {len(train):,} complaints. Test: {len(test):,} (most recent).",
             f"- **Product accuracy (TF-IDF + LR):** {accuracy_score(test['product'], pred):.3f} "
             f"(majority class: {(test['product'] == train['product'].mode()[0]).mean():.3f})",
             f"- **Monetary relief base rate in test:** {y.mean():.3f}"]
    if relief is not None:
        lines.append(f"- **Monetary relief (TF-IDF + LR, balanced):** {relief_scores(y, relief.predict(test['narrative']))}")
    lines += ["", "```", classification_report(test["product"], pred, zero_division=0), "```"]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:6]))


if __name__ == "__main__":
    main()
    from report.readme import update_quietly
    update_quietly()