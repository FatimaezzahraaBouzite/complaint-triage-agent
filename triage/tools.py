"""The agent's tools. All of them read REAL historical complaints (the 'train' split).

Guardrails live here, in code:
  * submit_triage only accepts labels that exist in the CFPB taxonomy
  * a priority floor is computed from risk signals (fraud, legal threat, hardship, vulnerable
    consumer) and the agent cannot file a complaint below it
  * routing to a team is a deterministic table, not an LLM choice
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

RISK_PATTERNS = {
    "fraud_or_identity_theft": r"identity theft|fraud|unauthori[sz]ed|stolen|scam",
    "legal_threat": r"attorney|lawyer|lawsuit|legal action|\bsue\b",
    "financial_hardship": r"foreclos|evict|repossess|bankruptcy|garnish",
}
TEAM_RULES = [("card", "Cards"), ("mortgage", "Mortgage servicing"), ("debt collection", "Collections"),
              ("money transfer", "Payments"), ("account", "Deposits"), ("loan", "Lending")]


def risk_flags(narrative, tags=""):
    flags = [name for name, pat in RISK_PATTERNS.items() if re.search(pat, narrative or "", re.I)]
    flags += [t for t, key in (("older_american", "Older American"), ("servicemember", "Servicemember")) if key in (tags or "")]
    return flags


def minimum_priority(flags):
    vulnerable = {"older_american", "servicemember"} & set(flags)
    if "financial_hardship" in flags or (vulnerable and len(flags) > len(vulnerable)):
        return "high"
    return "medium" if flags else "low"


def route(product, flags):
    if "fraud_or_identity_theft" in flags:
        return "Fraud & disputes"
    return next((team for key, team in TEAM_RULES if key in product.lower()), "General complaints")


class Toolbox:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.DB_PATH)
        con = sqlite3.connect(self.db_path)
        self.train = pd.read_sql("SELECT * FROM complaints WHERE split = 'train'", con)
        con.close()
        if self.train.empty:
            raise SystemExit("No complaints in the database. Run:  python -m triage.data")
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=80000,
                                   min_df=2 if len(self.train) > 2000 else 1)
        self.matrix = self.vec.fit_transform(self.train["narrative"])
        self.products = sorted(self.train["product"].unique())
        self.issues = {p: sorted(g["issue"].unique()) for p, g in self.train.groupby("product")}
        self.case, self.decision, self.rejections = None, None, 0

    def start_case(self, complaint: dict):
        self.case, self.decision, self.rejections = complaint, None, 0

    # ---- read tools -------------------------------------------------------
    def get_taxonomy(self, product=""):
        if not product:
            return {"products": self.train["product"].value_counts().to_dict()}
        if product not in self.issues:
            return {"error": f"Unknown product '{product}'.", "valid_products": self.products}
        counts = self.train.loc[self.train["product"] == product, "issue"].value_counts()
        return {"product": product, "issues": counts.to_dict()}

    def search_similar_complaints(self, query, product=""):
        sims = (self.vec.transform([query]) @ self.matrix.T).toarray()[0]
        if product in self.issues:
            sims = sims * (self.train["product"] == product).to_numpy()
        top = sims.argsort()[::-1][:5]
        rows = self.train.iloc[top]
        return {"results": [{"complaint_id": r.complaint_id, "similarity": round(float(s), 3), "product": r.product,
                             "issue": r.issue, "company": r.company, "actual_outcome": r.outcome,
                             "excerpt": r.narrative[:350]}
                            for r, s in zip(rows.itertuples(), sims[top]) if s > 0]}

    def get_outcome_stats(self, product, issue="", company=""):
        if product not in self.issues:
            return {"error": f"Unknown product '{product}'.", "valid_products": self.products}
        df, scope = self.train[self.train["product"] == product], {"product": product}
        for col, val in (("issue", issue), ("company", company)):
            if val:
                sub = df[df[col].str.lower() == val.lower()]
                if len(sub) >= 20:  # too few cases would give noisy rates: fall back to the wider group
                    df, scope[col] = sub, val
                else:
                    scope[f"{col}_ignored"] = f"only {len(sub)} past cases for '{val}'"
        return {"scope": scope, "past_cases": len(df),
                "outcome_rates": df["outcome"].value_counts(normalize=True).round(3).to_dict(),
                "timely_response_rate": round(float(df["timely"].mean()), 3)}

    # ---- the action -------------------------------------------------------
    def submit_triage(self, product, issue, priority, predicted_outcome, rationale, draft_reply):
        if self.case is None:
            return {"error": "No open complaint."}
        flags = risk_flags(self.case["narrative"], self.case.get("tags", ""))
        floor = minimum_priority(flags)
        problems = []
        if product not in self.issues:
            problems.append({"field": "product", "valid_values": self.products})
        elif issue not in self.issues[product]:
            problems.append({"field": "issue", "valid_values": self.issues[product]})
        if priority not in config.PRIORITIES:
            problems.append({"field": "priority", "valid_values": config.PRIORITIES})
        elif config.PRIORITIES.index(priority) < config.PRIORITIES.index(floor):
            problems.append({"field": "priority", "minimum_priority": floor, "because": flags})
        if predicted_outcome not in config.OUTCOMES.values():
            problems.append({"field": "predicted_outcome", "valid_values": sorted(config.OUTCOMES.values())})
        if not 40 <= len(draft_reply or "") <= 1500:
            problems.append({"field": "draft_reply", "rule": "between 40 and 1500 characters"})
        if problems:
            self.rejections += 1
            return {"error": "Triage rejected. Fix these fields and call submit_triage again.", "problems": problems}
        team = route(product, flags)
        self.decision = {"complaint_id": self.case.get("complaint_id", "manual"), "product": product, "issue": issue,
                         "priority": priority, "predicted_outcome": predicted_outcome, "team": team,
                         "risk_flags": flags, "rationale": rationale, "draft_reply": draft_reply}
        with closing(sqlite3.connect(self.db_path)) as con, con:  # closing(): Windows keeps the file locked otherwise
            con.execute("INSERT INTO triage_decisions (complaint_id, product, issue, priority, predicted_outcome, team, "
                        "risk_flags, rationale, draft_reply, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (*[json.dumps(v) if isinstance(v, list) else v for v in self.decision.values()],
                         datetime.now().isoformat(timespec="seconds")))
        return {"status": "saved", "routed_to": team, "risk_flags": flags}

    def run(self, name, args):
        if name not in TOOL_NAMES:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}


def _spec(name, description, required, optional=None):
    props = {**required, **(optional or {})}
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "required": list(required),
                             "properties": {k: {"type": "string", "description": v} for k, v in props.items()}}}


TOOL_SPECS = [
    _spec("search_similar_complaints", "Find the 5 most similar past complaints, with their product, issue and what the "
          "company actually did (actual_outcome). Use it to ground the classification and the outcome prediction.",
          {"query": "The complaint text, or its key sentences"}, {"product": "Optional exact product name to search within"}),
    _spec("get_taxonomy", "List valid product names (no argument) or the valid issues for one product. "
          "Labels passed to submit_triage must come from here.", {}, {"product": "Exact product name"}),
    _spec("get_outcome_stats", "Historical outcome rates (monetary relief / non-monetary relief / explanation) and timely-"
          "response rate for a product, optionally narrowed to an issue and a company.",
          {"product": "Exact product name"}, {"issue": "Exact issue name", "company": "Company name"}),
    _spec("submit_triage", "File the final triage decision. This ends the case. It is rejected if labels are not in the "
          "taxonomy or the priority is below the required minimum; fix and resubmit.",
          {"product": "Exact product name", "issue": "Exact issue name for that product", "priority": "low, medium or high",
           "predicted_outcome": "monetary_relief, non_monetary_relief or explanation",
           "rationale": "2-3 sentences: evidence for the labels, priority and predicted outcome",
           "draft_reply": "First reply to the consumer, under 120 words"}),
]
TOOL_NAMES = {t["name"] for t in TOOL_SPECS}
