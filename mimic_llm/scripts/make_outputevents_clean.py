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
    outputevents_path = os.path.join(ICU_DIR, "outputevents.csv.gz")
    items_path = os.path.join(ICU_DIR, "d_items.csv.gz")

    print("Reading outputevents from:", outputevents_path)
    print("Reading d_items from:", items_path)

    # 2. Read ICU item dictionary
    items = pd.read_csv(items_path, compression="gzip")

    # Keep only rows that belong to outputevents (if linksto exists)
    if "linksto" in items.columns:
        items = items[items["linksto"] == "outputevents"].copy()

    if "itemid" in items.columns:
        items = items.drop_duplicates(subset=["itemid"])

    # 3. Read outputevents with selected columns
    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "storetime",
        "value",
        "valueuom",
    ]

    outputevents = pd.read_csv(
        outputevents_path,
        compression="gzip",
        usecols=lambda c: c in usecols
    )

    print("Raw outputevents shape:", outputevents.shape)
    print("d_items (outputevents) shape:", items.shape)

    # 4. Merge to attach labels, normals, etc.
    merged = outputevents.merge(
        items,
        on="itemid",
        how="left",
        validate="m:1"
    )

    # 5. Compute a warning flag if we have normal ranges and numeric value
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
        # If we don't have the needed columns, create a neutral warning column
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
    output_clean = merged.drop(columns=existing_drop_cols)

    # 7. Convert charttime to datetime and add date/time columns
    if "charttime" in output_clean.columns:
        output_clean["charttime"] = pd.to_datetime(
            output_clean["charttime"], errors="coerce"
        )
        output_clean["date"] = output_clean["charttime"].dt.date
        output_clean["time"] = output_clean["charttime"].dt.time

    # 8. Prefix all non-ID columns with 'outputevents_'
    id_cols = ["subject_id", "hadm_id", "stay_id"]
    cols_to_rename = [c for c in output_clean.columns if c not in id_cols]

    rename_map = {col: "outputevents_" + col for col in cols_to_rename}
    output_clean = output_clean.rename(columns=rename_map)

    # 9. Save to processed folder
    out_path = os.path.join(ICU_PROC_DIR, "outputevents_clean.parquet")
    output_clean.to_parquet(out_path, index=False)

    print(f"Saved cleaned outputevents table to: {out_path}")
    print(f"Rows: {len(output_clean)}, Columns: {len(output_clean.columns)}")


if __name__ == "__main__":
    main()