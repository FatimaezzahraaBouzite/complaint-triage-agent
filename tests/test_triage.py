"""Run with:  pytest -q     (no network, no API key: uses tests/fixture.py and the KNN policy)"""
import sqlite3

import pandas as pd
import pytest

from triage import data
from triage.agent import TriageAgent
from triage.llm import KNNPolicy, LLMResponse
from triage.tools import Toolbox, minimum_priority, risk_flags, route

from .fixture import build_db, write_exports

CASE = {"complaint_id": "T1", "company": "ALPHA BANK", "state": "NY", "tags": "", "submitted_via": "Web",
        "narrative": "There is a charge on my credit card statement that I did not make and the bank will not remove it."}
GOOD = dict(product="Credit card", issue="Fees or interest", priority="low", predicted_outcome="explanation",
            rationale="Similar cases closed with explanation.", draft_reply="Thank you for contacting us. We are reviewing your complaint.")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_db(tmp_path_factory.mktemp("d") / "c.db")


@pytest.fixture
def box(db):
    b = Toolbox(db)
    b.start_case(dict(CASE))
    return b


# ---- data pipeline --------------------------------------------------------
def test_build_from_monthly_files(db):
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM complaints", con)
    con.close()
    assert set(df["split"]) == {"train", "test"}
    assert df.loc[df["split"] == "train", "date_received"].max() <= df.loc[df["split"] == "test", "date_received"].min()
    assert df["date_received"].min().startswith("2025-01") and df["date_received"].max().startswith("2026-04")  # all files read
    assert "In progress" not in set(df["company_response"]) and df["outcome"].notna().all()
    assert df["narrative"].str.len().min() >= 50 and df["complaint_id"].is_unique


def test_folder_reading(tmp_path):
    folder = write_exports(tmp_path / "dl", per_month=20)
    assert [p.name for p in data.find_files(folder)] == ["complaints_2025_Feb.csv", "complaints_2025_January.csv",
                                                         "complaints_2025_Mar.csv", "complaints_2026_Apr.csv"]
    pd.read_csv(folder / "complaints_2025_Feb.csv", dtype=str).to_excel(folder / "complaints_2025_Feb.xlsx", index=False)
    (folder / "complaints_2025_Feb.csv").unlink()                      # Excel files work as well
    pd.DataFrame({"a": [1]}).to_csv(folder / "complaints_bad.csv", index=False)  # wrong columns: skipped, not fatal
    raw = data.load_folder(folder)
    assert raw["Consumer complaint narrative"].notna().all() and raw["Date received"].str[:7].nunique() == 4
    with pytest.raises(SystemExit):
        data.find_files(tmp_path / "empty-does-not-exist")
    capped = data.normalize(raw, cap=5)
    assert capped.groupby("product").size().max() <= 5


def test_website_csv_headers_are_accepted():
    raw = pd.DataFrame({"Complaint ID": ["1", "2"], "Date received": ["2025-01-02", "2025-02-03"], "Product": ["Mortgage"] * 2,
                        "Issue": ["Trouble during payment process"] * 2, "Consumer complaint narrative": ["x" * 60, "y" * 60],
                        "Company": ["A", "B"], "Company response to consumer": ["Closed with explanation"] * 2,
                        "Timely response?": ["Yes", "No"]})
    df = data.normalize(raw)
    assert list(df["timely"]) == [1, 0] and list(df["outcome"]) == ["explanation"] * 2


# ---- tools ----------------------------------------------------------------
def test_search_returns_real_precedents_with_outcomes(box):
    hits = box.search_similar_complaints("collector keeps calling about a debt that is not mine")["results"]
    assert hits[0]["product"] == "Debt collection" and hits[0]["actual_outcome"] in {"explanation", "monetary_relief", "non_monetary_relief"}


def test_outcome_stats_and_taxonomy(box):
    s = box.get_outcome_stats("Credit card")
    assert s["past_cases"] > 20 and abs(sum(s["outcome_rates"].values()) - 1) < 0.01
    assert "error" in box.get_outcome_stats("Crypto")
    assert "Fees or interest" in box.get_taxonomy("Credit card")["issues"]


def test_risk_rules():
    assert minimum_priority(risk_flags("thanks for the info")) == "low"
    assert minimum_priority(risk_flags("this was an unauthorized charge")) == "medium"
    assert minimum_priority(risk_flags("unauthorized charge", "Older American")) == "high"
    assert minimum_priority(risk_flags("they started foreclosure")) == "high"
    assert route("Credit card", ["fraud_or_identity_theft"]) == "Fraud & disputes" and route("Mortgage", []) == "Mortgage servicing"


# ---- guardrails on the action --------------------------------------------
def test_submit_rejects_labels_outside_taxonomy(box):
    r = box.submit_triage(**{**GOOD, "product": "Credit cards!!"})
    assert "error" in r and box.decision is None
    r = box.submit_triage(**{**GOOD, "issue": "Managing an account"})  # valid issue, wrong product
    assert r["problems"][0]["field"] == "issue"


def test_submit_enforces_priority_floor(box):
    box.start_case({**CASE, "narrative": CASE["narrative"] + " This is fraud.", "tags": "Servicemember"})
    r = box.submit_triage(**GOOD)
    assert r["problems"][0]["minimum_priority"] == "high" and box.decision is None
    ok = box.submit_triage(**{**GOOD, "priority": "high"})
    assert ok["status"] == "saved" and ok["routed_to"] == "Fraud & disputes"
    con = sqlite3.connect(box.db_path)
    assert con.execute("SELECT COUNT(*) FROM triage_decisions WHERE complaint_id='T1'").fetchone()[0] == 1
    con.close()


# ---- agent ----------------------------------------------------------------
def test_agent_files_a_decision_end_to_end(db):
    res = TriageAgent(llm=KNNPolicy(), toolbox=Toolbox(db)).triage(dict(CASE))
    assert res.decision["product"] == "Credit card" and res.decision["team"] == "Cards"
    assert [t["name"] for t in res.tool_calls] == ["search_similar_complaints", "get_outcome_stats", "submit_triage"]


def test_agent_recovers_from_guardrail_rejection(db):
    case = {**CASE, "narrative": "They started foreclosure on my house while my mortgage modification was pending."}
    res = TriageAgent(llm=KNNPolicy(), toolbox=Toolbox(db)).triage(case)
    assert res.rejections == 1 and res.decision["priority"] == "high"


def test_agent_gives_up_safely(db):
    class Chatty:
        def complete(self, system, messages, tools):
            return LLMResponse(text="I think this is about a card.")
    res = TriageAgent(llm=Chatty(), toolbox=Toolbox(db)).triage(dict(CASE))
    assert res.decision is None and res.steps == 2  # one nudge, then handed to a human


def test_eval_and_analysis_run(db, tmp_path):
    from analysis import eda
    from evals import run_eval
    from triage import baseline
    s = run_eval.main(per_product=5, llm=KNNPolicy(), db_path=db, out_dir=tmp_path)
    assert s["complaints"] == 20 and s["filed_rate"] == 1.0 and s["guardrails"]["filed_below_priority_floor"] == 0
    assert s["agent"]["product_accuracy"] > 0.8
    eda.main(db, out=tmp_path)
    baseline.main(db, out=tmp_path / "b.md")
    assert "By product" in (tmp_path / "eda_report.md").read_text()
