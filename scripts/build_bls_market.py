"""
Consolidate the BLS Employment Projections flat-file database into a single
per-occupation market table keyed by SOC code.

Source (public, no key): https://download.bls.gov/pub/time.series/ep/
Requires a User-Agent with a contact email per BLS access policy.

The EP time-series DB stores one series per occupation×industry cell. The
occupation total across all industries is the row with ind_code == "TE1000";
only those rows carry the typical education / work-experience / training codes.
Value aspects were decoded against known Software Developers (15-1252) figures:

    A1  numeric employment change (thousands)
    A2  percent employment change over the projection decade   [growth %]
    A4  annual average occupational openings (thousands)
    A5  median annual wage (USD)
    PR  projected employment, target year (thousands)
    base employment = PR - A1

Output: data/raw/bls/occupation_market.csv (one row per SOC).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLS_DIR = ROOT / "data" / "raw" / "bls"
BASE_URL = "https://download.bls.gov/pub/time.series/ep/"
UA = os.getenv("RESEARCH_UA", "Hirely-research/1.0")

# Files needed from the EP flat-file DB
FILES = ["ep.series", "ep.aspect", "ep.eductrn", "ep.wkex", "ep.otjt", "ep.occupation"]

EDU_CODE = {
    "1": "Doctoral or professional degree", "2": "Master's degree",
    "3": "Bachelor's degree", "4": "Associate's degree",
    "5": "Postsecondary nondegree award", "6": "Some college, no degree",
    "7": "High school diploma or equivalent", "8": "No formal educational credential",
    "9": None,
}
# Note: avoid the bare label "None" — pandas.read_csv coerces it to NaN,
# which would erase the meaningful "no experience required" category.
WKEX_CODE = {"1": "5 years or more", "2": "Less than 5 years",
             "4": "No experience required", "5": None}
OTJT_CODE = {
    "1": "Internship/residency", "2": "Apprenticeship",
    "3": "Long-term on-the-job training", "4": "Moderate-term on-the-job training",
    "5": "Short-term on-the-job training", "6": "No on-the-job training", "7": None,
}


def download_flat_files() -> None:
    BLS_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = BLS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {name} ...", file=sys.stderr)
        req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as resp:
            dest.write_bytes(resp.read())


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(BLS_DIR / name, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def build() -> pd.DataFrame:
    ser = _read("ep.series")
    asp = _read("ep.aspect")

    # occupation totals across all industries carry the education/experience codes
    tot = ser[ser["ind_code"] == "TE1000"].copy()

    # pivot the value aspects we care about
    want = {"A1": "emp_change_k", "A2": "growth_pct", "A4": "openings_k",
            "A5": "median_wage", "PR": "proj_emp_k"}
    sub = asp[asp["aspect_type"].isin(want)].copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    wide = sub.pivot_table(index="series_id", columns="aspect_type",
                           values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns=want)

    m = tot.merge(wide, on="series_id", how="left")
    m["current_emp_k"] = m["proj_emp_k"] - m["emp_change_k"]
    m["typical_education"] = m["eductrn_code"].map(EDU_CODE)
    m["typical_experience"] = m["wkex_code"].map(WKEX_CODE)
    m["typical_training"] = m["otjt_code"].map(OTJT_CODE)

    out = m[[
        "occ_code", "typical_education", "typical_experience", "typical_training",
        "growth_pct", "openings_k", "median_wage", "current_emp_k", "proj_emp_k",
    ]].rename(columns={"occ_code": "soc_code"})
    # keep detailed occupations only (drop summary aggregates like 15-0000)
    out = out[~out["soc_code"].str.endswith("-0000")].reset_index(drop=True)
    return out


def main() -> None:
    download_flat_files()
    table = build()
    dest = BLS_DIR / "occupation_market.csv"
    table.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(table)} occupations)", file=sys.stderr)
    # sanity check
    sd = table[table["soc_code"] == "15-1252"]
    if not sd.empty:
        r = sd.iloc[0]
        print(f"  Software Developers: {r['typical_education']}, "
              f"+{r['growth_pct']}%, {r['openings_k']}K openings/yr, "
              f"${int(r['median_wage']):,} median", file=sys.stderr)


if __name__ == "__main__":
    main()
