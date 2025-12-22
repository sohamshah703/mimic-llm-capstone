import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import HOSP_PROC_DIR, COHORT_META_DIR, HOSP_PROC_COHORT_DIR


def main():
    cohort_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    dx_path = os.path.join(HOSP_PROC_DIR, "diagnoses_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading diagnoses from:", dx_path)

    cohort = pd.read_parquet(cohort_path)
    dx = pd.read_parquet(dx_path)

    hadm_ids = set(cohort["hadm_id"].unique())
    print("Number of cohort hadm_ids:", len(hadm_ids))

    dx_cohort = dx[dx["hadm_id"].isin(hadm_ids)].copy()

    out_path = os.path.join(HOSP_PROC_COHORT_DIR, "diagnoses_clean_icu_250.parquet")
    dx_cohort.to_parquet(out_path, index=False)

    print(f"Saved cohort-filtered diagnoses to: {out_path}")
    print(f"Rows: {len(dx_cohort)}, Cols: {len(dx_cohort.columns)}")


if __name__ == "__main__":
    main()