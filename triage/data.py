"""Build the project database from the CFPB's official complaint-narrative archive.

Since 14 August 2026 the CFPB no longer publishes consumer narratives in its live database, API or
exports. The narratives published before that date are in the CFPB FOIA reading room:
  https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
Download the "CCDB Export ..." zip files (no need to unzip them) and run:

  python -m triage.data --folder C:\\Users\\you\\Downloads    # reads the zips where they are
  python -m triage.data                                     # or copy them into data/raw/ first
  python -m triage.data --since 2022-01-01                  # use more history (default: 2024-01-01)
  python -m triage.data --per-product 5000                  # keep more complaints per product

Every file whose name starts with "ccdb" or "complaints" is tried (.zip, .csv, .json, .xlsx).
Archive files that end before --since are not opened. Files without a narrative column are skipped.
The archive is large (millions of rows), so rows are filtered while reading: only complaints with a
narrative, a closed outcome, an in-scope product and a date on or after --since are kept in memory.

The result is data/complaints.db with one table of real complaints and a time-based split:
  split = 'train'  history the agent may search (similar cases, outcome statistics)
  split = 'test'   the most recent complaints, held out for evaluation
"""
import argparse
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config

# our column -> accepted source headers, compared in lower case with punctuation and spaces removed,
# so "Consumer complaint narrative", "consumer_complaint_narrative" and "complaint_what_happened" all match.
ALIASES = {
    "complaint_id": ["complaintid"],
    "date_received": ["datereceived"],
    "product": ["product"],
    "sub_product": ["subproduct"],
    "issue": ["issue"],
    "sub_issue": ["subissue"],
    "narrative": ["consumercomplaintnarrative", "complaintwhathappened", "consumernarrative", "complaintnarrative", "narrative"],
    "company": ["company"],
    "state": ["state"],
    "tags": ["tags"],
    "submitted_via": ["submittedvia"],
    "company_response": ["companyresponsetoconsumer", "companyresponse"],
    "timely": ["timelyresponse", "timely"],
}
LOOKUP = {alias: ours for ours, names in ALIASES.items() for alias in names}
TABLE_EXT = {".csv", ".json", ".xlsx", ".xls"}
PREFIXES = ("ccdb", "complaints")
SEEN_PRODUCTS = Counter()  # product names met in narrative rows, for a helpful error message
BUILD_LOG = {}  # what load_folder saw, saved to build_summary.json for the README


def canon(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def rename_columns(df):
    """Rename source headers to our column names and drop every other column."""
    mapping = {}
    for c in df.columns:
        ours = LOOKUP.get(canon(c))
        if ours and ours not in mapping.values():
            mapping[c] = ours
    return df[list(mapping)].rename(columns=mapping)


def with_narrative(df):
    """Our columns only, rows that have a narrative only. None if the table has no narrative column."""
    df = rename_columns(df)
    if "narrative" not in df.columns:
        return None
    text = df["narrative"].astype("string").str.strip()
    return df[text.notna() & (text != "")]


def in_scope(df, since=None):
    """The row filters, applied chunk by chunk while reading so that memory stays small."""
    keep = df["narrative"].astype("string").str.strip().str.len().fillna(0) >= 50
    if "company_response" in df.columns:
        keep &= df["company_response"].isin(config.OUTCOMES)          # closed cases with a known outcome
    if "product" in df.columns:
        SEEN_PRODUCTS.update(df["product"].value_counts().to_dict())
        keep &= df["product"].isin(config.PRODUCTS)
    df = df[keep]
    if since is not None and "date_received" in df.columns and len(df):
        dates = pd.to_datetime(df["date_received"].astype(str).str.strip().str[:10], errors="coerce", format="mixed")
        df = df[dates >= since]
    return df


def read_table(name, opener, since=None):
    """Read one csv / json / excel table. `opener()` returns a fresh binary file object.
    Returns (rows in the table, rows with a narrative, DataFrame of kept rows or None, column names)."""
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        for encoding in ("utf-8-sig", "cp1252"):
            total, narr, kept, cols = 0, 0, [], []
            try:
                with opener() as f:
                    reader = pd.read_csv(f, dtype=str, chunksize=200_000, encoding=encoding,
                                         usecols=lambda c: canon(c) in LOOKUP)  # skip columns we never use
                    for chunk in reader:
                        total, cols = total + len(chunk), list(chunk.columns)
                        part = with_narrative(chunk)
                        if part is None:
                            return total, 0, None, cols
                        narr += len(part)
                        kept.append(in_scope(part, since))
                return total, narr, (pd.concat(kept, ignore_index=True) if kept else None), cols
            except UnicodeDecodeError:
                continue
        return 0, 0, None, []
    if ext == ".json":
        with opener() as f:
            head = f.read(1)
            f.seek(0)
            if head == b"[" or head == b"{":
                data = json.load(f)
                if isinstance(data, dict):  # {"hits": {"hits": [...]}} or {"data": [...]}
                    data = data.get("hits", {}).get("hits") if "hits" in data else next((v for v in data.values() if isinstance(v, list)), [])
                df = pd.DataFrame([r.get("_source", r) for r in data]).astype("string")
            else:
                df = pd.read_json(f, lines=True, dtype=False).astype("string")
    else:
        with opener() as f:
            df = pd.read_excel(io.BytesIO(f.read()), dtype=str)  # needs openpyxl
    part = with_narrative(df)
    return len(df), (0 if part is None else len(part)), (None if part is None else in_scope(part, since)), list(df.columns)


def read_file(path, since=None):
    """Yield (label, rows, narratives, frame_or_None, columns); a zip yields one entry per table inside it."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if Path(m).suffix.lower() in TABLE_EXT and not m.startswith("__MACOSX")]
            if not members:
                yield f"{path.name} (contains: {z.namelist()[:5]})", 0, 0, None, []
            for m in members:
                yield (path.name, *read_table(m, lambda m=m: z.open(m), since))
    else:
        yield (path.name, *read_table(path.name, lambda: open(path, "rb"), since))


def find_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower().startswith(PREFIXES)
                   and p.suffix.lower() in TABLE_EXT | {".zip"})
    if not files:
        names = ", ".join(sorted(p.name for p in folder.iterdir())[:15]) or "(empty)"
        raise SystemExit(f"No CCDB_Export*.zip or complaints* files in {folder}\nIt contains: {names}")
    return files


def ends_before(path, since):
    """True for archive files such as CCDB_Export_2_May_2018_through_April_2021.zip when --since is later."""
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", path.name)]
    return bool(years) and since is not None and max(years) < since.year


def load_folder(folder, since=None, per_file_cap=None):
    since = pd.Timestamp(since) if since else None
    files = find_files(folder)
    old = [p for p in files if ends_before(p, since)]
    files = [p for p in files if p not in old]
    print(f"Reading {len(files)} files from {Path(folder).resolve()}" + (f" (complaints since {since.date()})" if since else ""))
    if old:
        print(f"  Not opened, {len(old)} file(s) that end before {since.date()}: " + ", ".join(p.name for p in old[:3])
              + (" ..." if len(old) > 3 else ""))
    print(f"  {'file':<58} {'rows':>10} {'narratives':>11} {'kept':>8}")
    frames, skipped, read_log = [], [], []
    for path in files:
        try:
            for label, rows, narr, frame, cols in read_file(path, since):
                if frame is None:
                    skipped.append((label, cols))
                    continue
                print(f"  {label[:58]:<58} {rows:>10,} {narr:>11,} {len(frame):>8,}", flush=True)
                kept = len(frame)
                if per_file_cap and "product" in frame.columns:  # bound memory: sample each file per product
                    frame = frame.sample(frac=1, random_state=0).groupby("product").head(per_file_cap)
                read_log.append({"file": label, "rows": int(rows), "narratives": int(narr), "kept": int(kept),
                                 "after_file_sampling": int(len(frame))})
                frames.append(frame)
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            skipped.append((f"{path.name} (could not be read: {e})", []))
    if skipped:
        print(f"  Skipped {len(skipped)} file(s) with no narrative column: " + ", ".join(s[0] for s in skipped[:3])
              + (" ..." if len(skipped) > 3 else ""))
    if not frames:
        cols = next((c for _, c in skipped if c), [])
        raise SystemExit("None of the files contains complaint narratives.\n"
                         f"Columns found in the first skipped file: {cols}\n"
                         "The CFPB removed narratives from its live exports on 14 Aug 2026. Download the 'CCDB Export' "
                         "zip files from the CFPB narratives archive instead (link at the top of triage/data.py).")
    BUILD_LOG.clear()
    BUILD_LOG.update({"since": str(since.date()) if since is not None else None, "per_file_cap": per_file_cap,
                      "files_read": read_log, "files_not_opened": [p.name for p in old],
                      "files_skipped": [s_[0] for s_ in skipped]})
    out = pd.concat(frames, ignore_index=True)
    if out.empty and SEEN_PRODUCTS:
        raise SystemExit("Narratives were found, but none passed the filters (closed outcome, product in config.PRODUCTS, "
                         f"date since {since.date() if since else 'any'}).\nMost common product names in the files: "
                         + "; ".join(f"{k} ({v:,})" for k, v in SEEN_PRODUCTS.most_common(12)))
    return out


def normalize(raw: pd.DataFrame, cap=None) -> pd.DataFrame:
    df = rename_columns(raw).copy()
    for col in ALIASES:
        if col not in df.columns:
            df[col] = None
    required = [c for c in ("complaint_id", "date_received", "product", "issue", "company_response") if df[c].isna().all()]
    if required and len(df):
        raise SystemExit(f"The files have narratives but are missing these columns: {required}. Columns found: {list(raw.columns)}")
    steps = [("rows received from the files", len(df))]
    df["narrative"] = df["narrative"].fillna("").astype(str).str.strip()
    df = df[df["narrative"].str.len() >= 50]
    steps.append(("narrative of at least 50 characters", len(df)))
    df = df[df["company_response"].isin(config.OUTCOMES)]          # closed cases with a known outcome
    steps.append(("closed with a known outcome", len(df)))
    scoped = df["product"].isin(config.PRODUCTS)
    if scoped.any():
        df = df[scoped]
        steps.append(("in the products listed in config.PRODUCTS", len(df)))
    elif len(df):
        print("Note: no product matches config.PRODUCTS, keeping all products:", sorted(df["product"].dropna().unique()))
    df = df.drop_duplicates("complaint_id").copy()
    steps.append(("after removing duplicates", len(df)))
    if cap:  # keep the data set small enough for a laptop: random sample per product
        df = df.sample(frac=1, random_state=0).groupby("product").head(cap)
        steps.append((f"after capping at {cap:,} per product", len(df)))
    print("\n".join(f"  {n:>9,}  {label}" for label, n in steps))
    BUILD_LOG["normalize_steps"] = [{"label": label, "count": int(n)} for label, n in steps]
    dates = df["date_received"].astype(str).str.strip().str[:10]
    df["date_received"] = pd.to_datetime(dates, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    df["outcome"] = df["company_response"].map(config.OUTCOMES)
    df["timely"] = df["timely"].astype(str).str.lower().isin(["yes", "true"]).astype(int)
    for c in ("tags", "sub_product", "sub_issue", "state", "submitted_via", "company"):
        df[c] = df[c].fillna("").astype(str).replace({"None": "", "nan": "", "<NA>": ""})
    df = df.dropna(subset=["date_received", "product", "issue"]).sort_values(["date_received", "complaint_id"])
    cut = int(len(df) * (1 - config.TEST_FRACTION))
    df["split"] = ["train"] * cut + ["test"] * (len(df) - cut)
    df["complaint_id"] = df["complaint_id"].astype(str)
    return df[[*ALIASES, "outcome", "split"]].astype({c: str for c in ALIASES if c != "timely"}).reset_index(drop=True)


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
    write_summary(df, db_path, cap)
    return df


def fingerprint(db_path=None):
    """Short hash of the held-out complaint IDs. Two builds with the same fingerprint have the same test set,
    so evaluation results recorded on one are still valid for the other."""
    con = sqlite3.connect(db_path or config.DB_PATH)
    ids = [r[0] for r in con.execute("SELECT complaint_id FROM complaints WHERE split = 'test' ORDER BY complaint_id")]
    con.close()
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]


def write_summary(df, db_path, cap):
    """Save what the build did to build_summary.json, next to the database. The README is rendered from it."""
    split = {name: {"complaints": int(len(g)), "date_min": g["date_received"].min(), "date_max": g["date_received"].max()}
             for name, g in df.groupby("split")}
    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fingerprint(db_path),
        "per_product_cap": cap, "test_fraction": config.TEST_FRACTION, "products_in_scope": config.PRODUCTS,
        **{k: v for k, v in BUILD_LOG.items()},
        "complaints": int(len(df)), "products": int(df["product"].nunique()), "issues": int(df["issue"].nunique()),
        "companies": int(df["company"].nunique()), "date_min": df["date_received"].min(), "date_max": df["date_received"].max(),
        "split": split, "per_product": {k: int(v) for k, v in df.groupby("product").size().items()},
    }
    path = Path(db_path).with_name("build_summary.json")
    path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Build data/complaints.db from the CFPB narrative archive files.")
    p.add_argument("--folder", default=str(config.RAW_DIR), help="folder that holds the CCDB_Export*.zip files")
    p.add_argument("--since", default=config.DATE_MIN, help="keep complaints received on or after this date (YYYY-MM-DD)")
    p.add_argument("--per-product", type=int, default=3000, help="maximum complaints kept per product")
    a = p.parse_args()
    build(load_folder(a.folder, since=a.since, per_file_cap=a.per_product), cap=a.per_product)
    from report.readme import update_quietly
    update_quietly()


if __name__ == "__main__":
    main()