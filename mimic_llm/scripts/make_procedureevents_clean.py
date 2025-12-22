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
    procedureevents_path = os.path.join(ICU_DIR, "procedureevents.csv.gz")
    items_path = os.path.join(ICU_DIR, "d_items.csv.gz")

    print("Reading procedureevents from:", procedureevents_path)
    print("Reading d_items from:", items_path)

    # 2. Read ICU item dictionary
    items = pd.read_csv(items_path, compression="gzip")

    # Keep only rows that belong to procedureevents (if linksto exists)
    if "linksto" in items.columns:
        items = items[items["linksto"] == "procedureevents"].copy()

    if "itemid" in items.columns:
        items = items.drop_duplicates(subset=["itemid"])

    # 3. Read procedureevents with selected columns
    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "endtime",
        "storetime",
        "value",
        "valueuom",
        "location",
        "ordercategoryname",
        "ordercategorydescription",
        "ordercomponenttypedescription",
        "statusdescription",
    ]

    procedureevents = pd.read_csv(
        procedureevents_path,
        compression="gzip",
        usecols=lambda c: c in usecols
    )

    print("Raw procedureevents shape:", procedureevents.shape)
    print("d_items (procedureevents) shape:", items.shape)

    # 4. Merge to attach labels, normals, etc.
    merged = procedureevents.merge(
        items,
        on="itemid",
        how="left",
        validate="m:1"
    )

    # 5. Compute a warning flag if we have normal ranges and a numeric value
    if {"value", "lownormalvalue", "highnormalvalue"}.issubset(merged.columns):
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
        merged["lownormalvalue"] = pd.to_numeric(merged["lownormalvalue"], errors="coerce")
        merged["highnormalvalue"] = pd.to_numeric(merged["highnormalvalue"], errors="coerce")

        merged["warning"] = 0
        mask = (
            merged["value"].notna()
            & merged["lownormalvalue"].notna()
            & merged["highnormalvalue"].notna()
        )
        merged.loc[
            mask
            & ((merged["value"] < merged["lownormalvalue"])
               | (merged["value"] > merged["highnormalvalue"])),
            "warning"
        ] = 1
    else:
        merged["warning"] = 0

    # 6. Drop unnecessary columns
    cols_to_drop = [
        "storetime",
        "itemid",
        "lownormalvalue",
        "highnormalvalue",
        "abbreviation",
        "linksto",
        "param_type",
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in merged.columns]
    proc_clean = merged.drop(columns=existing_drop_cols)

    # 7. Convert starttime / endtime to datetime and add date columns
    for col in ["starttime", "endtime"]:
        if col in proc_clean.columns:
            proc_clean[col] = pd.to_datetime(proc_clean[col], errors="coerce")

    if "starttime" in proc_clean.columns:
        proc_clean["start_date"] = proc_clean["starttime"].dt.date
    if "endtime" in proc_clean.columns:
        proc_clean["end_date"] = proc_clean["endtime"].dt.date

    # Optionally drop raw datetime columns
    cols_to_drop2 = [c for c in ["starttime", "endtime"] if c in proc_clean.columns]
    proc_clean = proc_clean.drop(columns=cols_to_drop2)

    # 8. Prefix all non-ID columns with 'procedureevents_'
    id_cols = ["subject_id", "hadm_id", "stay_id"]
    cols_to_rename = [c for c in proc_clean.columns if c not in id_cols]

    rename_map = {col: "procedureevents_" + col for col in cols_to_rename}
    proc_clean = proc_clean.rename(columns=rename_map)

    # 9. Save to processed folder
    out_path = os.path.join(ICU_PROC_DIR, "procedureevents_clean.parquet")
    proc_clean.to_parquet(out_path, index=False)

    print(f"Saved cleaned procedureevents table to: {out_path}")
    print(f"Rows: {len(proc_clean)}, Columns: {len(proc_clean.columns)}")


if __name__ == "__main__":
    main()