import os
import sys
import pandas as pd

# --- Make sure Python can see paths.py in the project root ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../mimic_llm/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # .../mimic_llm

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import ICU_DIR, ICU_PROC_DIR


def main():
    # 1. Define input path
    icustays_path = os.path.join(ICU_DIR, "icustays.csv.gz")
    print("Reading icustays from:", icustays_path)

    # 2. Read raw icustays table
    df = pd.read_csv(icustays_path, compression="gzip")

    # 3. Convert intime and outtime to datetimes
    for col in ["intime", "outtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Create convenience date columns
    if "intime" in df.columns:
        df["icu_intake_date"] = df["intime"].dt.date
    if "outtime" in df.columns:
        df["icu_out_date"] = df["outtime"].dt.date

    # 5. Optionally ensure LOS is float (length of stay in days)
    if "los" in df.columns:
        df["los"] = pd.to_numeric(df["los"], errors="coerce")

    # 6. Save to processed folder
    out_path = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Saved cleaned icustays table to: {out_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()