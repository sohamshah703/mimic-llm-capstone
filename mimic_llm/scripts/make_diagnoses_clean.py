import os
import sys
import pandas as pd

# --- Make sure Python can see paths.py in the project root ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../mimic_llm/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # .../mimic_llm

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import HOSP_DIR, HOSP_PROC_DIR


def main():
    # 1. Define input paths for diagnoses and dictionary
    diag_path = os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz")
    diag_dict_path = os.path.join(HOSP_DIR, "d_icd_diagnoses.csv.gz")

    print("Reading diagnoses from:", diag_path)
    print("Reading diagnoses dictionary from:", diag_dict_path)

    # 2. Read raw tables
    diagnoses = pd.read_csv(diag_path, compression="gzip")
    diag_dict = pd.read_csv(diag_dict_path, compression="gzip")

    # 3. Optional: ensure dictionary has unique (icd_code, icd_version)
    if {"icd_code", "icd_version"}.issubset(diag_dict.columns):
        diag_dict = diag_dict.drop_duplicates(subset=["icd_code", "icd_version"])

    # 4. Merge to attach long_title (human-readable diagnosis)
    df = diagnoses.merge(
        diag_dict,
        on=["icd_code", "icd_version"],
        how="left",
        validate="m:1"  # many diagnoses rows to 1 dictionary row
    )

    # 5. Rename columns to make their purpose clear
    #    seq_num = diagnosis ordering within an admission
    if "seq_num" in df.columns:
        df = df.rename(columns={"seq_num": "dx_seq_num"})

    if "long_title" in df.columns:
        df = df.rename(columns={"long_title": "dx_long_title"})

    # 6. Optionally drop raw code columns (we have dx_long_title now)
    cols_to_drop = [c for c in ["icd_code", "icd_version"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 7. Save to processed folder as Parquet
    out_path = os.path.join(HOSP_PROC_DIR, "diagnoses_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Saved cleaned diagnoses table to: {out_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()