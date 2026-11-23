"""Configuration. Override with environment variables where shown."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("TRIAGE_DB", ROOT / "data" / "complaints.db"))
RAW_DIR = ROOT / "data" / "raw"   # the CCDB_Export*.zip files from the CFPB narratives archive
# Default start date. The CFPB renamed several products in 2023 (for example "Credit card or prepaid card"
# became "Credit card"); staying after that keeps a single taxonomy. Override with --since.
DATE_MIN = "2024-01-01"
TRACE_PATH = ROOT / "logs" / "traces.jsonl"

# Scope: the complaint desk of a bank / lender / payments company. Credit-bureau complaints
# (the majority of the database, largely templated dispute letters) are left out on purpose.
PRODUCTS = [
    "Credit card",
    "Checking or savings account",
    "Debt collection",
    "Mortgage",
    "Money transfer, virtual currency, or money service",
    "Student loan",
    "Vehicle loan or lease",
    "Payday loan, title loan, personal loan, or advance loan",
]

# What the company actually did (real label) -> the outcome class the agent predicts.
OUTCOMES = {"Closed with monetary relief": "monetary_relief",
            "Closed with non-monetary relief": "non_monetary_relief",
            "Closed with explanation": "explanation"}
PRIORITIES = ["low", "medium", "high"]
TEST_FRACTION = 0.15  # most recent complaints are held out: the agent never sees them

MODEL = os.getenv("AGENT_MODEL", "claude-haiku-5-5")
MAX_STEPS = 8