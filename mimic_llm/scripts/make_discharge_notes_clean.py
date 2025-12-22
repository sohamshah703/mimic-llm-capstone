import os
import sys
import pandas as pd

# --- Make sure Python can see paths.py in the project root ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../mimic_llm/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                   # .../mimic_llm

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import MIMIC_NOTES_DIR, NOTES_PROC_DIR


def main():
    # 1. Define input path
    discharge_path = os.path.join(MIMIC_NOTES_DIR, "discharge.csv.gz")
    print("Reading discharge notes from:", discharge_path)

    # 2. Read raw discharge notes
    df = pd.read_csv(discharge_path, compression="gzip")

    # 3. Convert time columns to datetime
    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Create convenience date columns (when note was charted)
    if "charttime" in df.columns:
        df["discharge_note_date"] = df["charttime"].dt.date

    # 5. (Optional) strip text whitespace
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).str.strip()

    # 6. Save to processed folder
    out_path = os.path.join(NOTES_PROC_DIR, "discharge_clean.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Saved cleaned discharge notes to: {out_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()