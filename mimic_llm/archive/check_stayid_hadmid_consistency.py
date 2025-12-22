#!/usr/bin/env python

import os
import sys
import pandas as pd

# --- Wire up project root / paths.py ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import ICU_PROC_DIR   # /scratch/.../mimic_proc_data/icu

def check_icustays_mapping():
    """
    Primary check: in icustays_clean.parquet, does any stay_id
    map to more than one hadm_id?

    Also prints basic counts:
    - total rows
    - unique stay_id
    - unique hadm_id
    - unique subject_id
    and a summary of hadm_id repetition.
    """
    icu_path = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    print(f"Reading icustays_clean from: {icu_path}")
    icu = pd.read_parquet(icu_path)

    # --- Basic stats ---
    total_rows        = len(icu)
    n_stay_ids        = icu["stay_id"].nunique()
    n_hadm_ids        = icu["hadm_id"].nunique()
    n_subject_ids     = icu["subject_id"].nunique()

    print("\n=== Basic stats for icustays_clean ===")
    print(f"Total rows              : {total_rows}")
    print(f"Unique stay_id          : {n_stay_ids}")
    print(f"Unique hadm_id          : {n_hadm_ids}")
    print(f"Unique subject_id       : {n_subject_ids}")

    # --- stay_id -> hadm_id mapping check ---
    counts_stay_to_hadm = icu.groupby("stay_id")["hadm_id"].nunique()
    bad_stay = counts_stay_to_hadm[counts_stay_to_hadm > 1].sort_values(ascending=False)

    print("\n=== stay_id → hadm_id mapping ===")
    print(f"Total unique stay_id                : {counts_stay_to_hadm.shape[0]}")
    print(f"stay_id with >1 hadm_id             : {bad_stay.shape[0]}")

    if bad_stay.empty:
        print("✅ All stay_id map to exactly one hadm_id in icustays_clean.")
    else:
        print("❌ Found stay_id with multiple hadm_id. Top few:")
        print(bad_stay.head())
        bad_ids = bad_stay.index.tolist()
        print("\nExample offending rows from icustays_clean:")
        print(
            icu[icu["stay_id"].isin(bad_ids)]
            .sort_values(["stay_id", "hadm_id"])
            .head(50)
        )

    # --- hadm_id repetition summary ---
    # Here each row is an ICU stay. So if unique hadm_id < total_rows,
    # some hadm_id appear in multiple rows (multiple ICU stays per admission).
    hadm_counts = icu.groupby("hadm_id")["stay_id"].nunique()
    hadm_with_multiple_stays = hadm_counts[hadm_counts > 1]

    print("\n=== hadm_id repetition (across rows / ICU stays) ===")
    print(f"hadm_id appearing in >1 row (ICU stays): {hadm_with_multiple_stays.shape[0]}")

    if not hadm_with_multiple_stays.empty:
        print("Example hadm_id with multiple ICU stays:")
        print(hadm_with_multiple_stays.sort_values(ascending=False).head())

def main():
    check_icustays_mapping()

if __name__ == "__main__":
    main()