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
    proc_path = os.path.join(ICU_PROC_DIR, "procedureevents_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading procedureevents from:", proc_path)

    cohort = pd.read_parquet(cohort_path)
    procs = pd.read_parquet(proc_path)

    stay_ids = set(cohort["stay_id"].unique())
    print("Number of cohort stay_ids:", len(stay_ids))

    procs_cohort = procs[procs["stay_id"].isin(stay_ids)].copy()

    out_path = os.path.join(ICU_PROC_COHORT_DIR, "procedureevents_clean_icu_250.parquet")
    procs_cohort.to_parquet(out_path, index=False)

    print(f"Saved cohort-filtered procedureevents to: {out_path}")
    print(f"Rows: {len(procs_cohort)}, Cols: {len(procs_cohort.columns)}")


if __name__ == "__main__":
    main()