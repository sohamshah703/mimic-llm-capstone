import os
import sys
import pandas as pd

# --- Make sure Python can see paths.py in the project root ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import ICU_DIR, ICU_PROC_DIR


def main():
    # 1. Define input paths
    inputevents_path = os.path.join(ICU_DIR, "inputevents.csv.gz")
    items_path = os.path.join(ICU_DIR, "d_items.csv.gz")

    print("Reading inputevents from:", inputevents_path)
    print("Reading d_items from:", items_path)

    # 2. Read ICU item dictionary
    items = pd.read_csv(items_path, compression="gzip")

    # Keep only rows that belong to inputevents (if linksto exists)
    if "linksto" in items.columns:
        items = items[items["linksto"] == "inputevents"].copy()

    # Ensure unique itemid in dictionary
    if "itemid" in items.columns:
        items = items.drop_duplicates(subset=["itemid"])

    # 3. Read inputevents with selected columns
    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "amount",
        "amountuom",
        "rate",
        "rateuom",
        "ordercategoryname",
        "ordercategorydescription",
        "ordercomponenttypedescription",
        "patientweight",
        "isopenbag",
        "originalamount",
        "originalamountuom",
        "originalrate",
        "originalrateuom",
    ]

    inputevents = pd.read_csv(
        inputevents_path,
        compression="gzip",
        usecols=lambda c: c in usecols
    )

    print("Raw inputevents shape:", inputevents.shape)
    print("d_items (inputevents) shape:", items.shape)

    # 4. Merge to attach item labels, category, units, etc.
    merged = inputevents.merge(
        items,
        on="itemid",
        how="left",        # keep ALL rows
        validate="m:1"
    )

    # 5. Drop unnecessary columns
    cols_to_drop = [
        "itemid",
        "abbreviation",
        "linksto",
        "isopenbag",
        "originalamount",
        "originalamountuom",
        "originalrate",
        "originalrateuom",
        "param_type",
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in merged.columns]
    medications = merged.drop(columns=existing_drop_cols)

    # 6. Convert starttime / endtime to datetime and add date/time splits
    for col in ["starttime", "endtime"]:
        if col in medications.columns:
            medications[col] = pd.to_datetime(medications[col], errors="coerce")

    if "starttime" in medications.columns:
        medications["start_date"] = medications["starttime"].dt.date
        medications["start_time"] = medications["starttime"].dt.time
    if "endtime" in medications.columns:
        medications["end_date"] = medications["endtime"].dt.date
        medications["end_time"] = medications["endtime"].dt.time

    # Optionally drop the original datetime columns if you want only derived ones
    # (okay by your "column drops allowed" rule)
    cols_to_drop2 = [c for c in ["starttime", "endtime"] if c in medications.columns]
    medications = medications.drop(columns=cols_to_drop2)

    # 7. Prefix all non-ID columns with 'medications_'
    id_cols = ["subject_id", "hadm_id", "stay_id"]
    cols_to_rename = [c for c in medications.columns if c not in id_cols]

    rename_map = {col: "medications_" + col for col in cols_to_rename}
    medications = medications.rename(columns=rename_map)

    # 8. Save to processed folder
    out_path = os.path.join(ICU_PROC_DIR, "medications_clean.parquet")
    medications.to_parquet(out_path, index=False)

    print(f"Saved cleaned medications table to: {out_path}")
    print(f"Rows: {len(medications)}, Columns: {len(medications.columns)}")


if __name__ == "__main__":
    main()