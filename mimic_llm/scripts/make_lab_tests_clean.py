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
    # 1. Define input paths
    labevents_path = os.path.join(HOSP_DIR, "labevents.csv.gz")
    labitems_path = os.path.join(HOSP_DIR, "d_labitems.csv.gz")

    print("Reading labevents from:", labevents_path)
    print("Reading d_labitems from:", labitems_path)

    # 2. Read labitems dictionary (small)
    labitems = pd.read_csv(labitems_path, compression="gzip")
    # ensure unique itemid in dictionary
    if "itemid" in labitems.columns:
        labitems = labitems.drop_duplicates(subset=["itemid"])

    # 3. Read labevents
    # NOTE: This is a very large table (~158M rows). This may use a lot of memory.
    # We read a subset of columns that we actually need, which is allowed
    # under your "dropping unnecessary columns is okay" rule.
    usecols = [
        "labevent_id",
        "subject_id",
        "hadm_id",
        "specimen_id",
        "itemid",
        "order_provider_id",
        "charttime",
        "storetime",
        "value",
        "valuenum",
        "valueuom",
        "ref_range_lower",
        "ref_range_upper",
        "flag",
        "priority",
        # if there are extra columns in labevents, they will just be ignored
    ]

    labevents = pd.read_csv(
        labevents_path,
        compression="gzip",
        usecols=lambda c: c in usecols  # keep the ones above, drop others if any
    )

    print("Raw labevents shape:", labevents.shape)
    print("Raw d_labitems shape:", labitems.shape)

    # 4. Merge labevents with labitems to attach labels, fluid, category, etc.
    merged = labevents.merge(
        labitems,
        on="itemid",
        how="left",        # keep ALL labevents rows
        validate="m:1"     # many labevents to 1 labitems row
    )

    # 5. Drop columns we don't need (column-level only, no row drops)
    cols_to_drop = [
        "ref_range_lower",
        "ref_range_upper",
        "priority",
        "order_provider_id",
        "storetime",
        "specimen_id",
        "labevent_id",
        "itemid",          # keep description from labitems instead
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in merged.columns]
    lab_tests = merged.drop(columns=existing_drop_cols)

    # 6. Convert charttime to datetime and add date/time columns
    if "charttime" in lab_tests.columns:
        lab_tests["charttime"] = pd.to_datetime(lab_tests["charttime"], errors="coerce")
        lab_tests["date"] = lab_tests["charttime"].dt.date
        lab_tests["time"] = lab_tests["charttime"].dt.time

    # 7. Rename 'flag' -> 'warning' and make it binary (1 = abnormal, 0 otherwise)
    if "flag" in lab_tests.columns:
        lab_tests = lab_tests.rename(columns={"flag": "warning"})
        # convert to string, handle NaN, compare case-insensitively
        lab_tests["warning"] = (
            lab_tests["warning"]
            .fillna("")
            .astype(str)
            .str.lower()
            .apply(lambda x: 1 if x == "abnormal" else 0)
        )

    # 8. Prefix all non-ID columns with 'lab_tests_' to avoid name clashes later
    #    We keep subject_id and hadm_id as-is.
    id_cols = ["subject_id", "hadm_id"]
    cols_to_rename = [c for c in lab_tests.columns if c not in id_cols]

    rename_map = {col: "lab_tests_" + col for col in cols_to_rename}
    lab_tests = lab_tests.rename(columns=rename_map)

    # 9. Save to processed folder as Parquet
    out_path = os.path.join(HOSP_PROC_DIR, "lab_tests_clean.parquet")
    lab_tests.to_parquet(out_path, index=False)

    print(f"Saved cleaned lab tests table to: {out_path}")
    print(f"Rows: {len(lab_tests)}, Columns: {len(lab_tests.columns)}")


if __name__ == "__main__":
    main()