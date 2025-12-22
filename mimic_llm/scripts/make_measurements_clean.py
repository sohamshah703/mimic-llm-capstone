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
    # 1. Define input paths
    chartevents_path = os.path.join(ICU_DIR, "chartevents.csv.gz")
    items_path = os.path.join(ICU_DIR, "d_items.csv.gz")

    print("Reading chartevents from:", chartevents_path)
    print("Reading d_items from:", items_path)

    # 2. Read ICU item dictionary
    items = pd.read_csv(items_path, compression="gzip")

    # Keep only rows that belong to chartevents (if linksto exists)
    if "linksto" in items.columns:
        items = items[items["linksto"] == "chartevents"].copy()

    # Ensure unique itemid in dictionary
    if "itemid" in items.columns:
        items = items.drop_duplicates(subset=["itemid"])

    # 3. Read chartevents (very large table) with selected columns only
    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "storetime",
        "value",
        "valuenum",
        "valueuom",
        "warning",   # existing warning flag in chartevents
    ]

    chartevents = pd.read_csv(
        chartevents_path,
        compression="gzip",
        usecols=lambda c: c in usecols  # keep only these columns
    )

    print("Raw chartevents shape:", chartevents.shape)
    print("d_items (chartevents) shape:", items.shape)

    # 4. Merge to attach labels, category, unitname, etc.
    merged = chartevents.merge(
        items,
        on="itemid",
        how="left",        # keep ALL chartevents rows
        validate="m:1"     # many chartevents to 1 dictionary row
    )

    # 5. Drop unnecessary columns (column-level only, no row drops)
    cols_to_drop = [
        "storetime",
        "itemid",
        "abbreviation",
        "linksto",
        "param_type",
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in merged.columns]
    measurements = merged.drop(columns=existing_drop_cols)

    # 6. Convert charttime to datetime and add date/time columns
    if "charttime" in measurements.columns:
        measurements["charttime"] = pd.to_datetime(
            measurements["charttime"], errors="coerce"
        )
        measurements["date"] = measurements["charttime"].dt.date
        measurements["time"] = measurements["charttime"].dt.time

    # 7. Prefix all non-ID columns with 'measurements_'
    id_cols = ["subject_id", "hadm_id", "stay_id"]
    cols_to_rename = [c for c in measurements.columns if c not in id_cols]

    rename_map = {col: "measurements_" + col for col in cols_to_rename}
    measurements = measurements.rename(columns=rename_map)

    # 8. Save to processed folder
    out_path = os.path.join(ICU_PROC_DIR, "measurements_clean.parquet")
    measurements.to_parquet(out_path, index=False)

    print(f"Saved cleaned measurements table to: {out_path}")
    print(f"Rows: {len(measurements)}, Columns: {len(measurements.columns)}")


if __name__ == "__main__":
    main()