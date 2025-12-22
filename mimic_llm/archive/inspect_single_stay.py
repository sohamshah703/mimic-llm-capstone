import os
import sys
import argparse
import pandas as pd

# --- wiring for imports from project root ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import (
    COHORT_META_DIR,
    ICU_PROC_COHORT_DIR,
    HOSP_PROC_COHORT_DIR,
    NOTES_PROC_COHORT_DIR,
)


def load_cohort():
    path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    cohort = pd.read_parquet(path)
    return cohort


def pretty_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect all data for a single ICU stay_id in the 250-stay cohort."
    )
    parser.add_argument(
        "--stay_id",
        type=int,
        default=None,
        help="ICU stay_id to inspect. If not provided, the first stay in the cohort is used.",
    )
    args = parser.parse_args()

    # 1. Load cohort and choose stay_id
    cohort = load_cohort()

    if args.stay_id is None:
        stay_id = int(cohort["stay_id"].iloc[0])
        print(f"No --stay_id provided, using first stay in cohort: {stay_id}")
    else:
        stay_id = args.stay_id
        print(f"Using provided stay_id: {stay_id}")

    if stay_id not in set(cohort["stay_id"].unique()):
        raise ValueError(f"stay_id {stay_id} not found in cohort_icu_250!")

    row = cohort[cohort["stay_id"] == stay_id].iloc[0]
    hadm_id = int(row["hadm_id"])
    subject_id = int(row["subject_id"])

    print(f"subject_id: {subject_id}")
    print(f"hadm_id   : {hadm_id}")
    print(f"stay_id   : {stay_id}")
    if "intime" in row and "outtime" in row:
        print(f"ICU intime: {row['intime']}")
        print(f"ICU outtime: {row['outtime']}")

    # 2. Load ICU tables (already filtered to cohort) and slice by stay_id
    pretty_header("ICU TABLES")

    def load_icu_table(filename):
        path = os.path.join(ICU_PROC_COHORT_DIR, filename)
        return pd.read_parquet(path)

    icu_cohort = load_icu_table("icustays_clean_icu_250.parquet")
    meas_cohort = load_icu_table("measurements_clean_icu_250.parquet")
    meds_cohort = load_icu_table("medications_clean_icu_250.parquet")
    out_cohort = load_icu_table("outputevents_clean_icu_250.parquet")
    proc_icu_cohort = load_icu_table("procedureevents_clean_icu_250.parquet")

    icu_this = icu_cohort[icu_cohort["stay_id"] == stay_id]
    meas_this = meas_cohort[meas_cohort["stay_id"] == stay_id]
    meds_this = meds_cohort[meds_cohort["stay_id"] == stay_id]
    out_this = out_cohort[out_cohort["stay_id"] == stay_id]
    proc_icu_this = proc_icu_cohort[proc_icu_cohort["stay_id"] == stay_id]

    print(f"icustays rows           : {len(icu_this)}")
    print(f"measurements rows       : {len(meas_this)}")
    print(f"medications rows        : {len(meds_this)}")
    print(f"outputevents rows       : {len(out_this)}")
    print(f"procedureevents (ICU)   : {len(proc_icu_this)}")

    print("\nicustays_clean (this stay):")
    print(icu_this.head())

    # 3. Load HOSP tables (cohort-filtered) and slice by hadm_id (and stay_id if present)
    pretty_header("HOSPITAL TABLES")

    def load_hosp_table(filename):
        path = os.path.join(HOSP_PROC_COHORT_DIR, filename)
        return pd.read_parquet(path)

    patadm_cohort = load_hosp_table("patients_admissions_clean_icu_250.parquet")
    dx_cohort = load_hosp_table("diagnoses_clean_icu_250.parquet")
    procs_cohort = load_hosp_table("procedures_clean_icu_250.parquet")
    labs_cohort = load_hosp_table("lab_tests_clean_icu_250.parquet")

    patadm_this = patadm_cohort[patadm_cohort["hadm_id"] == hadm_id]
    dx_this = dx_cohort[dx_cohort["hadm_id"] == hadm_id]

    # procedures/labs ICU-window tables *should* now also have stay_id after our filter step
    if "stay_id" in procs_cohort.columns:
        procs_this = procs_cohort[procs_cohort["stay_id"] == stay_id]
    else:
        procs_this = procs_cohort[procs_cohort["hadm_id"] == hadm_id]

    if "stay_id" in labs_cohort.columns:
        labs_this = labs_cohort[labs_cohort["stay_id"] == stay_id]
    else:
        labs_this = labs_cohort[labs_cohort["hadm_id"] == hadm_id]

    print(f"patients_admissions rows: {len(patadm_this)}")
    print(f"diagnoses rows          : {len(dx_this)}")
    print(f"procedures (ICU-window) : {len(procs_this)}")
    print(f"lab tests (ICU-window)  : {len(labs_this)}")

    print("\npatients_admissions (this hadm_id):")
    print(patadm_this.head())

    print("\ndiagnoses (first 10 rows):")
    print(dx_this.head(10))

    # 4. Load discharge note as target
    pretty_header("DISCHARGE NOTE (TARGET TEXT)")

    discharge_path = os.path.join(NOTES_PROC_COHORT_DIR, "discharge_clean_icu_250.parquet")
    discharge_cohort = pd.read_parquet(discharge_path)

    disc_this = discharge_cohort[discharge_cohort["hadm_id"] == hadm_id]

    print(f"discharge rows          : {len(disc_this)}")
    if len(disc_this) == 1:
        note_text = str(disc_this.iloc[0].get("text", ""))[:1500]
        print("\nDischarge note snippet (first 1500 chars):")
        print("-" * 80)
        print(note_text)
        print("-" * 80)
    else:
        print("WARNING: expected exactly 1 discharge note for this hadm_id.")

    pretty_header("SUMMARY COUNTS FOR THIS STAY")

    print(f"subject_id              : {subject_id}")
    print(f"hadm_id                 : {hadm_id}")
    print(f"stay_id                 : {stay_id}")
    print(f"icustays rows           : {len(icu_this)}")
    print(f"measurements rows       : {len(meas_this)}")
    print(f"medications rows        : {len(meds_this)}")
    print(f"outputevents rows       : {len(out_this)}")
    print(f"procedureevents (ICU)   : {len(proc_icu_this)}")
    print(f"patients_admissions rows: {len(patadm_this)}")
    print(f"diagnoses rows          : {len(dx_this)}")
    print(f"procedures (ICU-window) : {len(procs_this)}")
    print(f"lab tests (ICU-window)  : {len(labs_this)}")
    print(f"discharge notes rows    : {len(disc_this)}")


if __name__ == "__main__":
    main()