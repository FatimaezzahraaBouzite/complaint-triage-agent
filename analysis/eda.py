"""Analysis of the real complaints: what comes in, what companies do about it, and where triage time goes.

  python -m analysis.eda     # -> analysis/eda_report.md, analysis/relief_by_product.png
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from triage import config
from triage.tools import minimum_priority, risk_flags

OUT = Path(__file__).resolve().parent
READING_WPM = 238  # average adult silent reading speed, used to estimate analyst reading time


def md(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
           ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]


def pct(s):
    return (s * 100).round(1).astype(str) + "%"


def main(db_path=None, out=OUT):
    con = sqlite3.connect(db_path or config.DB_PATH)
    df = pd.read_sql("SELECT * FROM complaints", con)
    con.close()
    df["words"] = df["narrative"].str.split().str.len()
    df["money"] = df["outcome"].eq("monetary_relief")
    df["nonmoney"] = df["outcome"].eq("non_monetary_relief")
    df["floor"] = [minimum_priority(risk_flags(n, t)) for n, t in zip(df["narrative"], df["tags"])]

    def table(by, top=None):
        g = df.groupby(by).agg(complaints=("complaint_id", "size"), median_words=("words", "median"),
                               monetary_relief=("money", "mean"), non_monetary_relief=("nonmoney", "mean"),
                               timely=("timely", "mean")).sort_values("complaints", ascending=False)
        g = g.head(top) if top else g
        for c in ("monetary_relief", "non_monetary_relief", "timely"):
            g[c] = pct(g[c])
        g["median_words"] = g["median_words"].astype(int)
        return g

    flag_share = pd.Series({f: df["narrative"].str.contains(p, case=False, regex=True).mean()
                            for f, p in __import__("triage.tools", fromlist=["x"]).RISK_PATTERNS.items()})
    flag_share["older_american"] = df["tags"].str.contains("Older American").mean()
    flag_share["servicemember"] = df["tags"].str.contains("Servicemember").mean()
    read_hours = df["words"].sum() / READING_WPM / 60

    lines = ["# What the real complaints look like", "",
             "Source: CFPB Consumer Complaint Database (public). Complaints with a consumer narrative and a closed outcome.", "",
             f"- **Complaints:** {len(df):,} from {df['date_received'].min()} to {df['date_received'].max()}, "
             f"{df['company'].nunique():,} companies, {df['product'].nunique()} products, {df['issue'].nunique()} issues",
             f"- **Outcome mix:** " + ", ".join(f"{k} {v:.1%}" for k, v in df["outcome"].value_counts(normalize=True).items()),
             f"- **Narrative length:** median {int(df['words'].median())} words, 90th percentile {int(df['words'].quantile(.9))} words",
             f"- **Reading time alone:** {read_hours:,.0f} analyst-hours for this sample at {READING_WPM} words/min "
             f"({df['words'].mean() / READING_WPM:.1f} min per complaint before any research)",
             f"- **Priority floor from risk signals:** " + ", ".join(f"{k} {v:.1%}" for k, v in df["floor"].value_counts(normalize=True).items()),
             f"- **Risk signals (share of complaints):** " + ", ".join(f"{k} {v:.1%}" for k, v in flag_share.items()),
             "", "## By product", ""] + md(table("product")) + \
            ["", "## Top 12 issues", ""] + md(table("issue", 12)) + \
            ["", "## Top 10 companies", ""] + md(table("company", 10))
    (out / "eda_report.md").write_text("\n".join(lines) + "\n")

    g = df.groupby("product")[["money", "nonmoney"]].mean().sort_values("money")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(g.index.str.slice(0, 38), g["money"] * 100, color="#2a78d6", label="Monetary relief")
    ax.barh(g.index.str.slice(0, 38), g["nonmoney"] * 100, left=g["money"] * 100, color="#9cc3ee", label="Non-monetary relief")
    ax.set_xlabel("Share of complaints closed with relief (%)")
    ax.set_title("How often does the consumer get relief?")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "relief_by_product.png", dpi=150)
    print("\n".join(lines[4:10]))
    print(f"\nWrote {out / 'eda_report.md'} and relief_by_product.png")


if __name__ == "__main__":
    main()
