import os
import sys
import pandas as pd

# Folder where THIS script lives, e.g. /home/soham_shah/mimic_llm/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = parent folder of scripts, e.g. /home/soham_shah/mimic_llm
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import HOSP_DIR, HOSP_PROC_DIR

def main():
    # 1. Define input file paths
    admissions_path = os.path.join(HOSP_DIR, "admissions.csv.gz")
    patients_path = os.path.join(HOSP_DIR, "patients.csv.gz")

    print("Reading admissions from:", admissions_path)
    print("Reading patients from:", patients_path)

    # 2. Read raw tables
    admissions = pd.read_csv(admissions_path, compression="gzip")
    patients = pd.read_csv(patients_path, compression="gzip")

    # 3. Merge on subject_id (many admissions per patient)
    # validate="m:1" checks that patients has at most one row per subject_id
    df = admissions.merge(
        patients,
        on="subject_id",
        how="left",
        validate="m:1"
    )

    # 4. Convert date/time columns to proper datetimes
    datetime_cols = ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 5. Build a unified deathdate column using deathtime (admission) + dod (patients)
    #    -> first, ensure dod is datetime as well
    if "dod" in df.columns:
        df["dod"] = pd.to_datetime(df["dod"], errors="coerce")

    # start with deathtime as deathdate
    df["deathdate"] = df.get("deathtime")

    # where deathdate is missing, fill from dod
    if "dod" in df.columns:
        missing_death = df["deathdate"].isna() & df["dod"].notna()
        df.loc[missing_death, "deathdate"] = df.loc[missing_death, "dod"]

    # 6. Ensure hospital_expire_flag is 1 whenever deathdate is present
    if "hospital_expire_flag" in df.columns:
        df.loc[df["deathdate"].notna(), "hospital_expire_flag"] = 1

    # 7. Create simple date-only columns for convenience
    if "admittime" in df.columns:
        df["admit_date"] = df["admittime"].dt.date
    if "dischtime" in df.columns:
        df["discharge_date"] = df["dischtime"].dt.date

    # 8. Drop columns we don’t plan to use (can adjust later if needed)
    drop_cols = [
        "anchor_year",
        "anchor_year_group",
        "admit_provider_id",
        "insurance",
        "language",
        "marital_status",
        "race",
        "dod",          # we now have deathdate instead
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # 9. Save to processed folder as Parquet
    out_path = os.path.join(HOSP_PROC_DIR, "patients_admissions_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Saved cleaned table to: {out_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()