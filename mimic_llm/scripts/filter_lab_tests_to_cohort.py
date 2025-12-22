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
    labs_path = os.path.join(HOSP_PROC_DIR, "lab_tests_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading lab_tests from:", labs_path)

    cohort = pd.read_parquet(cohort_path)
    labs = pd.read_parquet(labs_path)

    cohort_small = cohort[["hadm_id", "stay_id", "intime", "outtime"]].copy()
    cohort_small["intime"] = pd.to_datetime(cohort_small["intime"], errors="coerce")
    cohort_small["outtime"] = pd.to_datetime(cohort_small["outtime"], errors="coerce")

    labs["lab_tests_charttime"] = pd.to_datetime(
        labs["lab_tests_charttime"], errors="coerce"
    )

    # Join labs to cohort by hadm_id
    merged = labs.merge(
        cohort_small,
        on="hadm_id",
        how="inner",
        validate="m:m"
    )

    # Filter to ICU window
    mask = (
        merged["lab_tests_charttime"] >= merged["intime"]
    ) & (
        merged["lab_tests_charttime"] <= merged["outtime"]
    )

    labs_window = merged[mask].copy()

    out_path = os.path.join(HOSP_PROC_COHORT_DIR, "lab_tests_clean_icu_250.parquet")
    labs_window.to_parquet(out_path, index=False)

    print(f"Saved ICU-window lab tests to: {out_path}")
    print(f"Rows: {len(labs_window)}, Cols: {len(labs_window.columns)}")


if __name__ == "__main__":
    main()