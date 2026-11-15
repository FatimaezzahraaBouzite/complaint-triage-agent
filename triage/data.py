"""Build the project database from the monthly complaint files exported from the CFPB website.

Put the exported files (complaints_2025_January, complaints_2025_Feb, ...) in data/raw/, or point at
the folder they are in:

  python -m triage.data                                   # reads data/raw/
  python -m triage.data --folder C:\\Users\\you\\Downloads    # reads the files where they are
  python -m triage.data --per-product 3000                # keep more complaints per product

Any file whose name starts with "complaints" and ends with .csv, .xlsx or .xls is read.

The result is data/complaints.db with one table of real complaints and a time-based split:
  split = 'train'  history the agent may search (similar cases, outcome statistics)
  split = 'test'   the most recent complaints, held out for evaluation
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from . import config

# our column -> header in the CFPB export (a lower-case API-style name is accepted too)
ALIASES = {
    "complaint_id": ["Complaint ID", "complaint_id"],
    "date_received": ["Date received", "date_received"],
    "product": ["Product", "product"],
    "sub_product": ["Sub-product", "sub_product"],
    "issue": ["Issue", "issue"],
    "sub_issue": ["Sub-issue", "sub_issue"],
    "narrative": ["Consumer complaint narrative", "complaint_what_happened"],
    "company": ["Company", "company"],
    "state": ["State", "state"],
    "tags": ["Tags", "tags"],
    "submitted_via": ["Submitted via", "submitted_via"],
    "company_response": ["Company response to consumer", "company_response"],
    "timely": ["Timely response?", "timely"],
}
EXTENSIONS = {".csv", ".xlsx", ".xls"}


def find_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.name.lower().startswith("complaints") and p.suffix.lower() in EXTENSIONS)
    if not files:
        names = ", ".join(sorted(p.name for p in folder.iterdir())[:15]) or "(empty)"
        raise SystemExit(f"No complaints*.csv or complaints*.xlsx files in {folder}\nIt contains: {names}")
    return files


def read_file(path):
    """Read one export and keep only the rows that have a consumer narrative."""
    if path.suffix.lower() == ".csv":
        try:
            chunks = list(pd.read_csv(path, dtype=str, chunksize=100_000, encoding="utf-8-sig"))
        except UnicodeDecodeError:  # file was re-saved from Excel with a Windows encoding
            chunks = list(pd.read_csv(path, dtype=str, chunksize=100_000, encoding="cp1252"))
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    else:
        df = pd.read_excel(path, dtype=str)  # needs openpyxl
    df.columns = [str(c).strip() for c in df.columns]
    col = next((c for c in ALIASES["narrative"] if c in df.columns), None)
    if col is None:
        print(f"  SKIPPED {path.name}: no 'Consumer complaint narrative' column. Columns: {list(df.columns)[:8]}...")
        return None
    kept = df[df[col].notna()]
    print(f"  {path.name:<34} {len(df):>8,} rows  {len(kept):>8,} with a narrative")
    return kept


def load_folder(folder):
    files = find_files(folder)
    print(f"Reading {len(files)} files from {Path(folder).resolve()}")
    frames = [f for f in (read_file(p) for p in files) if f is not None]
    if not frames:
        raise SystemExit("None of the files could be used.")
    return pd.concat(frames, ignore_index=True)


def normalize(raw: pd.DataFrame, cap=None) -> pd.DataFrame:
    df = pd.DataFrame({ours: next((raw[c] for c in names if c in raw.columns), None) for ours, names in ALIASES.items()})
    steps = [("rows with a narrative, all files", len(df))]
    df["narrative"] = df["narrative"].fillna("").astype(str).str.strip()
    df = df[df["narrative"].str.len() >= 50]
    steps.append(("narrative of at least 50 characters", len(df)))
    df = df[df["company_response"].isin(config.OUTCOMES)]          # closed cases with a known outcome
    steps.append(("closed with a known outcome", len(df)))
    in_scope = df["product"].isin(config.PRODUCTS)
    if in_scope.any():
        df = df[in_scope]
        steps.append(("in the products listed in config.PRODUCTS", len(df)))
    elif len(df):
        print("Note: no product matches config.PRODUCTS, keeping all products:", sorted(df["product"].dropna().unique()))
    df = df.drop_duplicates("complaint_id").copy()
    steps.append(("after removing duplicates", len(df)))
    if cap:  # keep the data set small enough for a laptop: random sample per product
        df = df.sample(frac=1, random_state=0).groupby("product").head(cap)
        steps.append((f"after capping at {cap:,} per product", len(df)))
    print("\n".join(f"  {n:>9,}  {label}" for label, n in steps))
    df["outcome"] = df["company_response"].map(config.OUTCOMES)
    dates = df["date_received"].astype(str).str.strip().str[:10]
    df["date_received"] = pd.to_datetime(dates, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    df["timely"] = df["timely"].astype(str).str.lower().isin(["yes", "true"]).astype(int)
    for c in ("tags", "sub_product", "sub_issue", "state", "submitted_via", "company"):
        df[c] = df[c].fillna("").astype(str).replace({"None": "", "nan": ""})
    df = df.dropna(subset=["date_received", "product", "issue"]).sort_values(["date_received", "complaint_id"])
    cut = int(len(df) * (1 - config.TEST_FRACTION))
    df["split"] = ["train"] * cut + ["test"] * (len(df) - cut)
    df["complaint_id"] = df["complaint_id"].astype(str)
    return df.reset_index(drop=True)


def build(raw: pd.DataFrame, db_path=None, cap=None):
    db_path = Path(db_path or config.DB_PATH)
    df = normalize(raw, cap)
    if df.empty:
        raise SystemExit("No usable complaints after cleaning. The counts above show which filter removed them.")
    db_path.parent.mkdir(exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    df.to_sql("complaints", con, index=False)
    con.execute("CREATE INDEX idx_split ON complaints(split)")
    con.execute("""CREATE TABLE triage_decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_id TEXT, product TEXT,
                   issue TEXT, priority TEXT, predicted_outcome TEXT, team TEXT, risk_flags TEXT, rationale TEXT,
                   draft_reply TEXT, created_at TEXT)""")
    con.commit()
    con.close()
    t = df[df["split"] == "test"]
    print(f"Built {db_path}: {len(df):,} complaints, {df['product'].nunique()} products, {df['issue'].nunique()} issues, "
          f"{df['company'].nunique():,} companies, {df['date_received'].min()} to {df['date_received'].max()}")
    print(f"Held-out test set: {len(t):,} complaints received on or after {t['date_received'].min()}")
    print(df.groupby("product").size().rename("complaints").to_string())
    return df


def main():
    p = argparse.ArgumentParser(description="Build data/complaints.db from exported CFPB complaint files.")
    p.add_argument("--folder", default=str(config.RAW_DIR), help="folder that holds the complaints_* files")
    p.add_argument("--per-product", type=int, default=2000, help="maximum complaints kept per product")
    a = p.parse_args()
    build(load_folder(a.folder), cap=a.per_product)


if __name__ == "__main__":
    main()
