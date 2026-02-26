import json
import pandas as pd
from pathlib import Path
import re

# ============================
# CONFIG
# ============================

DATA_PATH = Path("data/raw/CUAD_v1.json")
OUTPUT_PATH = Path("data/processed/cuad_processed.csv")

# Map CUAD clause types to your 5 risk categories
label_mapping = {
    # Liability
    "Cap On Liability": "Liability Risk",
    "Uncapped Liability": "Liability Risk",
    "Liquidated Damages": "Liability Risk",

    # Termination
    "Termination For Convenience": "Termination Risk",
    "Notice Period To Terminate Renewal": "Termination Risk",
    "Renewal Term": "Termination Risk",
    "Expiration Date": "Termination Risk",

    # Payment
    "Revenue/Profit Sharing": "Payment Risk",
    "Minimum Commitment": "Payment Risk",
    "Price Restrictions": "Payment Risk",
    "Volume Restriction": "Payment Risk",

    # Indemnity-like legal protection
    "Insurance": "Indemnity Risk",

    # Standard / Low Risk
    "Governing Law": "Standard Clause",
    "Assignment": "Standard Clause",
    "Third Party Beneficiary": "Standard Clause",
}

# ============================
# LOAD JSON
# ============================

print("Loading CUAD_v1.json...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

contracts = data["data"]
print(f"Total contracts found: {len(contracts)}")

# ============================
# EXTRACT CLAUSES
# ============================

rows = []

for contract in contracts:
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            question_text = qa["question"]

            # Extract clause type inside quotes
            match = re.search(r'"([^"]+)"', question_text)
            if not match:
                continue

            clause_type = match.group(1)

            if clause_type in label_mapping and not qa["is_impossible"]:
                for answer in qa["answers"]:
                    clause_text = answer["text"].strip()

                    if len(clause_text) > 30:
                        rows.append({
                            "clause_text": clause_text,
                            "risk_category": label_mapping[clause_type]
                        })

df = pd.DataFrame(rows)

if len(df) == 0:
    print("No clauses extracted. Check mapping.")
    exit()

df = df.drop_duplicates()

print("\nTotal extracted clauses:", len(df))
print("\nClass distribution:\n")
print(df["risk_category"].value_counts())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved dataset to {OUTPUT_PATH}")