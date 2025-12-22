#!/usr/bin/env python

"""
check_cohort_discharge_consistency.py

Checks that every hadm_id used in cohort_icu_250.parquet has
EXACTLY ONE discharge summary in discharge_clean.parquet.

Also prints a few summary stats and explicitly verifies the
three forced stay_ids.
"""

import os
import sys
import pandas as pd

# --- wire up project root and paths.py ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import COHORT_META_DIR, NOTES_PROC_DIR  # type: ignore

FORCED_STAY_IDS = [38657298, 35527336, 35517464]


def main():
    cohort_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    discharge_path = os.path.join(NOTES_PROC_DIR, "discharge_clean.parquet")

    print("Reading cohort from      :", cohort_path)
    print("Reading discharge from   :", discharge_path)

    cohort = pd.read_parquet(cohort_path)
    discharge = pd.read_parquet(discharge_path)

    if "hadm_id" not in cohort.columns:
        raise ValueError("cohort_icu_250.parquet must contain 'hadm_id' column.")
    if "hadm_id" not in discharge.columns or "note_id" not in discharge.columns:
        raise ValueError("discharge_clean.parquet must contain 'hadm_id' and 'note_id' columns.")

    # --- basic cohort stats ---
    n_stays = cohort["stay_id"].nunique()
    n_hadm  = cohort["hadm_id"].nunique()
    print(f"\nCohort summary:")
    print(f"- Unique stay_id in cohort : {n_stays}")
    print(f"- Unique hadm_id in cohort : {n_hadm}")

    cohort_hadm_ids = set(cohort["hadm_id"].unique())

    # --- restrict discharge to only those hadm_id present in cohort ---
    disc_cohort = discharge[discharge["hadm_id"].isin(cohort_hadm_ids)].copy()

    # counts of discharge notes per hadm_id within the cohort
    notes_per_hadm = (
        disc_cohort.groupby("hadm_id")["note_id"]
        .nunique()
        .rename("n_discharge_notes")
    )

    n_with_notes = len(notes_per_hadm)
    n_without_notes = len(cohort_hadm_ids) - n_with_notes

    print("\nDischarge note counts for cohort hadm_id:")
    print(f"- hadm_id with ≥1 discharge note : {n_with_notes}")
    print(f"- hadm_id with 0 discharge note  : {n_without_notes}")

    # Split by exact count
    hadm_eq_1 = notes_per_hadm[notes_per_hadm == 1]
    hadm_gt_1 = notes_per_hadm[notes_per_hadm > 1]

    print(f"- hadm_id with exactly 1 note    : {len(hadm_eq_1)}")
    print(f"- hadm_id with >1 notes          : {len(hadm_gt_1)}")

    if n_without_notes == 0 and len(hadm_gt_1) == 0:
        print("\n✅ All hadm_id in cohort_icu_250 have EXACTLY ONE discharge summary.")
    else:
        print("\n❌ Inconsistency detected:")
        if n_without_notes > 0:
            print(f"  - {n_without_notes} cohort hadm_id have 0 discharge notes.")
        if len(hadm_gt_1) > 0:
            print(f"  - {len(hadm_gt_1)} cohort hadm_id have >1 discharge notes.")
            print("    Example offending hadm_id → count:")
            print(hadm_gt_1.head(10).to_string())

    # --- Check the forced stay_ids explicitly ---
    print("\nForced stay_ids check:")
    missing_forced = [s for s in FORCED_STAY_IDS if s not in set(cohort['stay_id'])]
    if missing_forced:
        print(f"❌ These forced stay_id are NOT present in cohort: {missing_forced}")
    else:
        print(f"✅ All forced stay_ids are present in cohort: {FORCED_STAY_IDS}")

    # Map forced stay_ids → hadm_id and their discharge note counts
    forced_rows = cohort[cohort["stay_id"].isin(FORCED_STAY_IDS)].copy()
    for s in FORCED_STAY_IDS:
        rows = forced_rows[forced_rows["stay_id"] == s]
        if rows.empty:
            continue
        hadm = int(rows.iloc[0]["hadm_id"])
        n_notes = int(notes_per_hadm.get(hadm, 0))
        print(f"- stay_id {s} → hadm_id {hadm}, discharge note count: {n_notes}")

    print("\nDone.")


if __name__ == "__main__":
    main()