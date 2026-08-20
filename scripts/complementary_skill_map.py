"""Build a complementary software-skill pair map from ONET data.

This script reads the ONET software skills table in datasets/onet and ranks
software-skill pairs that co-occur in demand-heavy occupation profiles.
The output is saved as a ranked CSV file for additional analysis.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "onet"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "complementary_skill_map"
OUTPUT_FILE = OUTPUT_DIR / "complementary_skill_pairs.csv"
SUMMARY_FILE = OUTPUT_DIR / "complementary_skill_pairs_summary.md"


def load_skill_frame(file_name: str) -> pd.DataFrame:
    """Load an ONET skill file and keep only the importance rows we need."""
    path = DATA_DIR / file_name
    df = pd.read_excel(path)

    if file_name == "Software Skills.xlsx":
        hot = df["Hot Technology"].astype(str).str.strip().eq("Y") if "Hot Technology" in df.columns else pd.Series(False, index=df.index)
        in_demand = df["In Demand"].astype(str).str.strip().eq("Y") if "In Demand" in df.columns else pd.Series(False, index=df.index)
        df = df[hot | in_demand].copy()

    if "O*NET-SOC Code" in df.columns:
        df = df.rename(columns={"O*NET-SOC Code": "occupation_code"})
    if "Title" in df.columns:
        df = df.rename(columns={"Title": "occupation_title"})
    if "Element Name" in df.columns:
        df = df.rename(columns={"Element Name": "skill_name"})
    if "Data Value" in df.columns:
        df = df.rename(columns={"Data Value": "importance"})
    if "importance" not in df.columns:
        df["importance"] = 1.0

    df["skill_name"] = df["skill_name"].astype(str).str.strip()
    df["occupation_code"] = df["occupation_code"].astype(str).str.strip()
    df["occupation_title"] = df["occupation_title"].astype(str).str.strip()

    return df[["occupation_code", "occupation_title", "skill_name", "importance"]].dropna(subset=["skill_name"])


def build_occupation_skill_table() -> pd.DataFrame:
    """Combine skill categories into one occupation → skill table."""
    files = ["Software Skills.xlsx"]

    frames = [load_skill_frame(name) for name in files]
    combined = pd.concat(frames, ignore_index=True)

    combined = combined.drop_duplicates(subset=["occupation_code", "skill_name"], keep="first")

    skill_counts = combined.groupby("occupation_code")["skill_name"].nunique().reset_index(name="skill_count")
    combined = combined.merge(skill_counts, on="occupation_code", how="left")
    combined = combined[combined["skill_count"] >= 2].copy()

    return combined.drop(columns=["skill_count"])


def compute_pair_scores(occupation_skill_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pair support, lift and a simple synergy score."""
    occupation_skill_df = occupation_skill_df.copy()

    occupation_sets = (
        occupation_skill_df.groupby("occupation_code", sort=False)
        .agg(skills=("skill_name", lambda s: tuple(sorted(set(s)))) )
        .reset_index()
    )

    total_occupations = len(occupation_sets)
    pair_counts = {}
    skill_counts = {}

    for _, row in occupation_sets.iterrows():
        skills = row["skills"]
        for a, b in itertools.combinations(skills, 2):
            key = tuple(sorted((a, b)))
            pair_counts[key] = pair_counts.get(key, 0) + 1

        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    rows = []
    for (skill_a, skill_b), support in pair_counts.items():
        freq_a = skill_counts.get(skill_a, 0)
        freq_b = skill_counts.get(skill_b, 0)

        expected_support = (freq_a * freq_b) / total_occupations
        lift = support / expected_support if expected_support > 0 else 0.0
        synergy = support * lift

        rows.append({
            "skill_a": skill_a,
            "skill_b": skill_b,
            "support_occupations": support,
            "support_rate": support / total_occupations,
            "freq_a": freq_a,
            "freq_b": freq_b,
            "lift": lift,
            "synergy_score": synergy,
        })

    pair_df = pd.DataFrame(rows)
    pair_df = pair_df[(pair_df["support_occupations"] >= 2) &
                      (pair_df["freq_a"] >= 5) &
                      (pair_df["freq_b"] >= 5)].copy()
    pair_df = pair_df.sort_values(["synergy_score", "lift"], ascending=[False, False]).reset_index(drop=True)
    return pair_df


def main() -> None:
    occupation_skill_df = build_occupation_skill_table()
    pair_df = compute_pair_scores(occupation_skill_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_df.to_csv(OUTPUT_FILE, index=False)

    summary = [
        "# Complementary software-skill map",
        "",
        "This map ranks software-skill pairs that co-occur in high-demand ONET occupation profiles.",
        "It uses co-occurrence frequency, lift, and a simple synergy score to highlight pairs that are stronger together than either skill alone.",
        "",
        "## Top 20 pairs",
        "",
    ]

    top_pairs = pair_df.head(20).copy()
    for _, row in top_pairs.iterrows():
        summary.append(
            f"- {row['skill_a']} + {row['skill_b']} — support={int(row['support_occupations'])}, "
            f"lift={row['lift']:.3f}, synergy={row['synergy_score']:.1f}"
        )

    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Built {len(pair_df)} complementary skill pairs.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")

    print("\nTop 25 complementary skill pairs:")
    print(pair_df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
