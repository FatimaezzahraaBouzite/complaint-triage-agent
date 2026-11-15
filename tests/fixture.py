"""Writes a few small monthly files in the same format as the CFPB website export, with made-up narratives.
It exists only so the tests run without the real files. It is NOT real data and is never used by the app."""
import random

import pandas as pd

from triage import data

SPEC = {  # product -> issue -> sentence fragments
    "Credit card": {
        "Problem with a purchase shown on your statement": ["a charge on my credit card statement that I did not make", "the merchant billed my card twice"],
        "Fees or interest": ["a late fee and interest were added to my card even though I paid on time", "the annual fee was charged after I closed the card"]},
    "Mortgage": {
        "Trouble during payment process": ["my mortgage payment was applied to the wrong month by the servicer", "the escrow amount on my home loan doubled without notice"],
        "Struggling to pay mortgage": ["I asked for a loan modification and now they threaten foreclosure on my house", "the servicer ignored my forbearance request for the mortgage"]},
    "Debt collection": {
        "Attempts to collect debt not owed": ["a collector keeps calling about a debt that is not mine", "the collection agency reported a debt I already paid"],
        "Communication tactics": ["the debt collector calls my workplace many times a day", "the collector used abusive language on the phone"]},
    "Checking or savings account": {
        "Managing an account": ["the bank froze my checking account and I cannot access my deposit", "an overdraft fee was taken from my savings account in error"],
        "Closing an account": ["the bank closed my account and has not returned my funds", "I asked to close my checking account and they keep charging fees"]},
}
EXTRAS = ["", "", " This is fraud and I reported identity theft.", " I will contact my attorney.", " I have called XXXX times since XX/XX/XXXX."]
RESPONSES = ["Closed with explanation"] * 7 + ["Closed with monetary relief"] * 2 + ["Closed with non-monetary relief", "In progress"]
MONTHS = [("2025_January", "2025-01"), ("2025_Feb", "2025-02"), ("2025_Mar", "2025-03"), ("2026_Apr", "2026-04")]


def month_rows(prefix, n, rng, start_id):
    rows = []
    for i in range(n):
        product = rng.choice(list(SPEC))
        issue = rng.choice(list(SPEC[product]))
        text = f"I am writing because {rng.choice(SPEC[product][issue])}. I contacted the company and nothing was fixed.{rng.choice(EXTRAS)}"
        rows.append({
            "Date received": f"{prefix}-{1 + i % 28:02d}", "Product": product, "Sub-product": "", "Issue": issue, "Sub-issue": "",
            "Consumer complaint narrative": text if rng.random() < 0.9 else None, "Company public response": "",
            "Company": rng.choice(["ALPHA BANK", "BETA FINANCIAL", "GAMMA SERVICING"]), "State": rng.choice(["NY", "TX", "CA"]),
            "ZIP code": "10001", "Tags": rng.choice([None, None, None, "Older American", "Servicemember"]),
            "Consumer consent provided?": "Consent provided", "Submitted via": "Web", "Date sent to company": f"{prefix}-28",
            "Company response to consumer": rng.choice(RESPONSES), "Timely response?": rng.choice(["Yes"] * 9 + ["No"]),
            "Consumer disputed?": "N/A", "Complaint ID": str(start_id + i)})
    return rows


def write_exports(folder, per_month=95, seed=3):
    """Monthly CSV files named like the real downloads: complaints_2025_January.csv, ..."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for k, (name, prefix) in enumerate(MONTHS):
        pd.DataFrame(month_rows(prefix, per_month, rng, 9000000 + k * 1000)).to_csv(folder / f"complaints_{name}.csv", index=False)
    (folder / "notes.txt").write_text("not a complaints file")
    return folder


def build_db(path):
    raw_dir = write_exports(path.parent / "raw")
    data.build(data.load_folder(raw_dir), db_path=path)
    return path
