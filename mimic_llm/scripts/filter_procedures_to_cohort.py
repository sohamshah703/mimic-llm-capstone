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
    proc_path = os.path.join(HOSP_PROC_DIR, "procedures_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading procedures from:", proc_path)

    cohort = pd.read_parquet(cohort_path)
    procs = pd.read_parquet(proc_path)

    # Keep only needed columns from cohort
    cohort_small = cohort[["hadm_id", "stay_id", "intime", "outtime"]].copy()

    # Ensure times are datetime
    cohort_small["intime"] = pd.to_datetime(cohort_small["intime"], errors="coerce")
    cohort_small["outtime"] = pd.to_datetime(cohort_small["outtime"], errors="coerce")

    procs["procedure_chartdatetime"] = pd.to_datetime(
        procs["procedure_chartdatetime"], errors="coerce"
    )

    # Join procedures to cohort by hadm_id to get ICU windows + stay_id
    merged = procs.merge(
        cohort_small,
        on="hadm_id",
        how="inner",          # keep only admissions in cohort
        validate="m:m"
    )

    # Filter to ICU window
    mask = (
        merged["procedure_chartdatetime"] >= merged["intime"]
    ) & (
        merged["procedure_chartdatetime"] <= merged["outtime"]
    )

    procs_window = merged[mask].copy()

    out_path = os.path.join(HOSP_PROC_COHORT_DIR, "procedures_clean_icu_250.parquet")
    procs_window.to_parquet(out_path, index=False)

    print(f"Saved ICU-window procedures to: {out_path}")
    print(f"Rows: {len(procs_window)}, Cols: {len(procs_window.columns)}")


if __name__ == "__main__":
    main()