import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import NOTES_PROC_DIR, COHORT_META_DIR, NOTES_PROC_COHORT_DIR


def main():
    cohort_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    discharge_path = os.path.join(NOTES_PROC_DIR, "discharge_clean.parquet")

    print("Reading cohort from:", cohort_path)
    print("Reading discharge notes from:", discharge_path)

    cohort = pd.read_parquet(cohort_path)
    discharge = pd.read_parquet(discharge_path)

    hadm_ids = set(cohort["hadm_id"].unique())
    print("Number of cohort hadm_ids:", len(hadm_ids))

    df = discharge[discharge["hadm_id"].isin(hadm_ids)].copy()

    # Ensure charttime exists and is datetime
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        # sort so we can pick the latest note per hadm_id
        df = df.sort_values(["hadm_id", "charttime"])
        df_latest = df.groupby("hadm_id").tail(1).copy()
    else:
        # If no charttime, just pick the last row per hadm_id by index
        df_latest = df.sort_values(["hadm_id"]).groupby("hadm_id").tail(1).copy()

    out_path = os.path.join(NOTES_PROC_COHORT_DIR, "discharge_clean_icu_250.parquet")
    df_latest.to_parquet(out_path, index=False)

    print(f"Saved cohort-filtered discharge notes to: {out_path}")
    print(f"Rows: {len(df_latest)}, Cols: {len(df_latest.columns)}")


if __name__ == "__main__":
    main()