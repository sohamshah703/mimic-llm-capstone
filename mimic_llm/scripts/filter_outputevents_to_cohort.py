import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import ICU_PROC_DIR, COHORT_META_DIR, ICU_PROC_COHORT_DIR


def main():
    cohort_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    out_path_full = os.path.join(ICU_PROC_DIR, "outputevents_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading outputevents from:", out_path_full)

    cohort = pd.read_parquet(cohort_path)
    outputs = pd.read_parquet(out_path_full)

    stay_ids = set(cohort["stay_id"].unique())
    print("Number of cohort stay_ids:", len(stay_ids))

    outputs_cohort = outputs[outputs["stay_id"].isin(stay_ids)].copy()

    out_path = os.path.join(ICU_PROC_COHORT_DIR, "outputevents_clean_icu_250.parquet")
    outputs_cohort.to_parquet(out_path, index=False)

    print(f"Saved cohort-filtered outputevents to: {out_path}")
    print(f"Rows: {len(outputs_cohort)}, Cols: {len(outputs_cohort.columns)}")


if __name__ == "__main__":
    main()