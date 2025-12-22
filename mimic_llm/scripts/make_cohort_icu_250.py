#!/usr/bin/env python

"""
make_cohort_icu_250.py

Build a 250-stay ICU cohort with the following rules:

1. Only include ICU stays whose corresponding hadm_id has a discharge
   summary, and specifically where the number of discharge summaries == 1.
2. Hard-include the following 3 stay_ids in the cohort:
   - 38657298
   - 35527336
   - 35517464
3. Fill the remaining slots with a random sample from the rest.

Output:
- Writes cohort_icu_250.parquet to COHORT_META_DIR.
"""

import os
import sys
import pandas as pd

# --- Wire up project root and paths.py ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import ICU_PROC_DIR, NOTES_PROC_DIR, COHORT_META_DIR  # type: ignore


FORCED_STAY_IDS = [38657298, 35527336, 35517464]
TARGET_COHORT_SIZE = 250
RANDOM_SEED = 42


def main():
    # 1. Load full ICU stays table (processed, not yet cohort-filtered)
    icu_path = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    print(f"Reading ICU stays from: {icu_path}")
    icu = pd.read_parquet(icu_path)

    if "stay_id" not in icu.columns or "hadm_id" not in icu.columns:
        raise ValueError("icustays_clean.parquet must contain 'stay_id' and 'hadm_id' columns.")

    # 2. Load full discharge notes table
    disc_path = os.path.join(NOTES_PROC_DIR, "discharge_clean.parquet")
    print(f"Reading discharge notes from: {disc_path}")
    disc = pd.read_parquet(disc_path)

    if "hadm_id" not in disc.columns or "note_id" not in disc.columns:
        raise ValueError("discharge_clean.parquet must contain 'hadm_id' and 'note_id' columns.")

    # 3. Keep only hadm_id with exactly 1 discharge note
    notes_per_hadm = (
        disc.groupby("hadm_id")["note_id"]
        .nunique()
        .rename("n_discharge_notes")
    )

    print("\n=== Discharge summary counts per hadm_id ===")
    print(f"Total hadm_id with any discharge note      : {len(notes_per_hadm)}")
    print(f"Max discharge notes per hadm_id            : {notes_per_hadm.max()}")

    hadm_with_exactly_one_note = notes_per_hadm[notes_per_hadm == 1].index
    print(f"hadm_id with exactly 1 discharge note      : {len(hadm_with_exactly_one_note)}")

    # 4. Filter ICU stays to only those admissions with exactly 1 discharge note
    icu_filtered = icu[icu["hadm_id"].isin(hadm_with_exactly_one_note)].copy()

    print("\n=== ICU stays after filtering by discharge summaries ===")
    print(f"Total ICU stays before filter              : {icu['stay_id'].nunique()}")
    print(f"ICU stays with hadm_id having 1 discharge  : {icu_filtered['stay_id'].nunique()}")

    # 5. Ensure forced stay_ids are present and satisfy the filter
    all_stay_ids = set(icu["stay_id"].unique())
    missing_forced = [s for s in FORCED_STAY_IDS if s not in all_stay_ids]

    if missing_forced:
        raise ValueError(
            f"The following forced stay_id(s) do not exist in icustays_clean: {missing_forced}"
        )

    forced_rows = icu_filtered[icu_filtered["stay_id"].isin(FORCED_STAY_IDS)].copy()
    missing_in_filtered = sorted(set(FORCED_STAY_IDS) - set(forced_rows["stay_id"]))

    if missing_in_filtered:
        # This means those stays exist, but their hadm_id does NOT have exactly 1 discharge note.
        # Since your requirement is "only hadm_id with discharge summaries (=1)", we stop here.
        raise ValueError(
            "The following forced stay_id(s) do not belong to admissions with exactly 1 "
            f"discharge summary and therefore cannot be included under the current rules: "
            f"{missing_in_filtered}"
        )

    print("\n=== Forced stays check ===")
    print(f"Forced stay_ids                            : {FORCED_STAY_IDS}")
    print(f"Forced stay_ids present after filter       : {sorted(forced_rows['stay_id'].unique().tolist())}")

    n_forced = len(forced_rows)
    if n_forced != len(FORCED_STAY_IDS):
        raise ValueError(
            f"Expected {len(FORCED_STAY_IDS)} forced stays in filtered set, "
            f"but found {n_forced}."
        )

    # 6. Sample the remaining stays to reach TARGET_COHORT_SIZE
    n_to_sample = TARGET_COHORT_SIZE - n_forced
    print(f"\nWe need to sample {n_to_sample} additional stays to reach {TARGET_COHORT_SIZE}.")

    remaining = icu_filtered[~icu_filtered["stay_id"].isin(FORCED_STAY_IDS)].copy()
    n_available = remaining["stay_id"].nunique()

    print(f"Available non-forced stays after filter    : {n_available}")

    if n_available < n_to_sample:
        raise ValueError(
            f"Not enough remaining stays to sample {n_to_sample}. "
            f"Only {n_available} available after applying filters."
        )

    sampled = (
        remaining
        .drop_duplicates(subset=["stay_id"])
        .sample(n=n_to_sample, random_state=RANDOM_SEED)
        .copy()
    )

    # 7. Build final cohort: forced rows + sampled rows
    cohort = pd.concat([forced_rows, sampled], ignore_index=True)

    # Optional: enforce one row per stay_id
    cohort = cohort.drop_duplicates(subset=["stay_id"]).copy()

    print("\n=== Final cohort summary ===")
    print(f"Total rows in cohort                       : {len(cohort)}")
    print(f"Unique stay_id in cohort                   : {cohort['stay_id'].nunique()}")
    print(f"Unique hadm_id in cohort                   : {cohort['hadm_id'].nunique()}")

    # Sort for readability (by stay_id)
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)

    # 8. Save to COHORT_META_DIR
    os.makedirs(COHORT_META_DIR, exist_ok=True)
    out_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    cohort.to_parquet(out_path, index=False)

    print(f"\nCohort saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()