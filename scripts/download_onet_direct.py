#!/usr/bin/env python3
"""Download selected O*NET Excel files directly from db_30_3_excel folder.
"""
import os
from urllib.parse import quote
import requests

BASE = "https://www.onetcenter.org/dl_files/database/db_30_3_excel/"
FILES = [
    "Software Skills.xlsx",
    "Essential Skills.xlsx",
    "Transferable Skills.xlsx",
    "Knowledge.xlsx",
    "Abilities.xlsx",
    "Education.xlsx",
    "Training and Experience.xlsx",
    "Work Activities.xlsx",
    "Work Styles.xlsx",
    "Occupation Data.xlsx",
    "Job Titles.xlsx",
    "Task Statements.xlsx",
    "Content Model Reference.xlsx",
    "Occupation Level Metadata.xlsx",
]

OUT_DIR = "datasets/onet"

os.makedirs(OUT_DIR, exist_ok=True)

for fname in FILES:
    url = BASE + quote(fname)
    target = os.path.join(OUT_DIR, fname)
    print(f"Downloading {fname}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(target, "wb") as f:
            f.write(r.content)
        print(f"Saved -> {target}")
    except Exception as e:
        print(f"Failed to download {fname}: {e}")

print("Done.")
