#!/usr/bin/env python

"""
Explore processed HOSP / ICU / NOTES tables in MIMIC:

Answers (or helps answer) Q1–Q10 from Soham's list.

Run with:

    cd ~/mimic_llm
    micromamba activate mimic-llm
    python scripts/explore_mimic_proc_stats.py

WARNING: Some sections (labs, measurements) load very large tables.
Run this on a compute node, not login, and comment out sections if needed.
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Wire up project root and paths.py
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import HOSP_PROC_DIR, ICU_PROC_DIR, NOTES_PROC_DIR


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------
# Q1 – Diagnoses per hospital admission
# ---------------------------------------------------------------------
def q1_diagnoses():
    """
    1. For 546,028 hospital admissions:
       - Does every hadm_id have at least 1 diagnosis?
       - Or are there admissions with 0 diagnoses?
    """

    print_section("Q1: Diagnoses per hospital admission")

    adm_path = os.path.join(HOSP_PROC_DIR, "patients_admissions_clean.parquet")
    dx_path  = os.path.join(HOSP_PROC_DIR, "diagnoses_clean.parquet")

    adm = pd.read_parquet(adm_path, columns=["subject_id", "hadm_id"])
    dx  = pd.read_parquet(dx_path,  columns=["subject_id", "hadm_id"])

    n_hadm_total = adm["hadm_id"].nunique()
    n_dx_rows    = len(dx)

    # Which admissions have at least one diagnosis?
    hadm_with_dx = dx["hadm_id"].unique()
    hadm_all     = adm["hadm_id"].unique()

    hadm_without_dx = np.setdiff1d(hadm_all, hadm_with_dx)

    n_with_dx    = hadm_with_dx.shape[0]
    n_without_dx = hadm_without_dx.shape[0]

    print(f"Total unique hadm_id (hospital admissions): {n_hadm_total}")
    print(f"Total diagnoses rows                       : {n_dx_rows}")
    print(f"Average diagnoses per admission            : {n_dx_rows / n_hadm_total:0.2f}")
    print()
    print(f"Admissions with ≥1 diagnosis               : {n_with_dx} "
          f"({100.0 * n_with_dx / n_hadm_total:0.2f}%)")
    print(f"Admissions with 0 diagnoses                : {n_without_dx} "
          f"({100.0 * n_without_dx / n_hadm_total:0.2f}%)")

    if n_without_dx > 0:
        print("\nExample hadm_id with NO diagnoses:")
        print(hadm_without_dx[:10])


# ---------------------------------------------------------------------
# Q2 – Procedures (HOSP) per hospital admission
# ---------------------------------------------------------------------
def q2_hosp_procedures():
    """
    2. For 859,655 procedures total:
       - Does every hadm_id have at least 1 hospital procedure?
       - Or are there admissions with 0 procedures?
    """

    print_section("Q2: HOSP Procedures per hospital admission")

    adm_path  = os.path.join(HOSP_PROC_DIR, "patients_admissions_clean.parquet")
    proc_path = os.path.join(HOSP_PROC_DIR, "procedures_clean.parquet")

    adm  = pd.read_parquet(adm_path,  columns=["subject_id", "hadm_id"])
    proc = pd.read_parquet(proc_path, columns=["subject_id", "hadm_id"])

    n_hadm_total = adm["hadm_id"].nunique()
    n_proc_rows  = len(proc)

    hadm_with_proc = proc["hadm_id"].unique()
    hadm_all       = adm["hadm_id"].unique()
    hadm_without_proc = np.setdiff1d(hadm_all, hadm_with_proc)

    n_with_proc    = hadm_with_proc.shape[0]
    n_without_proc = hadm_without_proc.shape[0]

    print(f"Total unique hadm_id (hospital admissions): {n_hadm_total}")
    print(f"Total HOSP procedures rows                 : {n_proc_rows}")
    print(f"Average HOSP procedures per admission      : {n_proc_rows / n_hadm_total:0.2f}")
    print()
    print(f"Admissions with ≥1 HOSP procedure          : {n_with_proc} "
          f"({100.0 * n_with_proc / n_hadm_total:0.2f}%)")
    print(f"Admissions with 0 HOSP procedures          : {n_without_proc} "
          f"({100.0 * n_without_proc / n_hadm_total:0.2f}%)")

    if n_without_proc > 0:
        print("\nExample hadm_id with NO HOSP procedures:")
        print(hadm_without_proc[:10])


# ---------------------------------------------------------------------
# Q3 – Lab tests per hospital admission + unique labels/fluids/categories
# ---------------------------------------------------------------------
def q3_lab_tests():
    """
    3. For 158,374,764 lab tests:
       - Does every hadm_id have at least 1 lab test?
       - Or are there admissions with 0 labs?
       - Count unique lab_tests_label, lab_tests_fluid, lab_tests_category
       - Show 5 example values from each.
    """

    print_section("Q3: Lab tests per hospital admission and lab meta")

    adm_path = os.path.join(HOSP_PROC_DIR, "patients_admissions_clean.parquet")
    labs_path = os.path.join(HOSP_PROC_DIR, "lab_tests_clean.parquet")

    adm = pd.read_parquet(adm_path, columns=["subject_id", "hadm_id"])
    # Load only the columns we need from labs (still a huge table!)
    labs = pd.read_parquet(
        labs_path,
        columns=[
            "subject_id",
            "hadm_id",
            "lab_tests_label",
            "lab_tests_fluid",
            "lab_tests_category",
        ],
    )

    n_hadm_total = adm["hadm_id"].nunique()
    n_lab_rows   = len(labs)

    hadm_with_lab = labs["hadm_id"].unique()
    hadm_all      = adm["hadm_id"].unique()

    hadm_without_lab = np.setdiff1d(hadm_all, hadm_with_lab)

    n_with_lab    = hadm_with_lab.shape[0]
    n_without_lab = hadm_without_lab.shape[0]

    print(f"Total unique hadm_id (hospital admissions): {n_hadm_total}")
    print(f"Total lab test rows                        : {n_lab_rows}")
    print(f"Average lab tests per admission            : {n_lab_rows / n_hadm_total:0.2f}")
    print()
    print(f"Admissions with ≥1 lab test                : {n_with_lab} "
          f"({100.0 * n_with_lab / n_hadm_total:0.2f}%)")
    print(f"Admissions with 0 lab tests                : {n_without_lab} "
          f"({100.0 * n_without_lab / n_hadm_total:0.2f}%)")

    # Unique meta fields
    print("\nUnique counts for lab meta columns:")
    print(f"- lab_tests_label    : {labs['lab_tests_label'].nunique()}")
    print(f"- lab_tests_fluid    : {labs['lab_tests_fluid'].nunique()}")
    print(f"- lab_tests_category : {labs['lab_tests_category'].nunique()}")

    print("\nExample lab_tests_label (5):")
    print(labs["lab_tests_label"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample lab_tests_fluid (5):")
    print(labs["lab_tests_fluid"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample lab_tests_category (5):")
    print(labs["lab_tests_category"].dropna().drop_duplicates().head(5).tolist())


# ---------------------------------------------------------------------
# Q4 – ICU stays vs hospital admissions (just a quick check)
# ---------------------------------------------------------------------
def q4_icu_vs_hosp():
    """
    4. Confirm that not all hospital admissions have an ICU stay.
    """

    print_section("Q4: ICU stays vs hospital admissions")

    adm_path = os.path.join(HOSP_PROC_DIR, "patients_admissions_clean.parquet")
    icu_path = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")

    adm = pd.read_parquet(adm_path, columns=["subject_id", "hadm_id"])
    icu = pd.read_parquet(icu_path, columns=["subject_id", "hadm_id", "stay_id"])

    n_hadm_total   = adm["hadm_id"].nunique()
    n_stay_total   = icu["stay_id"].nunique()
    hadm_with_icu  = icu["hadm_id"].unique()
    hadm_without_icu = np.setdiff1d(adm["hadm_id"].unique(), hadm_with_icu)

    print(f"Total unique hadm_id (hospital admissions): {n_hadm_total}")
    print(f"Total unique stay_id (ICU stays)          : {n_stay_total}")
    print(f"Admissions with ≥1 ICU stay               : {len(hadm_with_icu)} "
          f"({100.0 * len(hadm_with_icu) / n_hadm_total:0.2f}%)")
    print(f"Admissions with 0 ICU stays               : {len(hadm_without_icu)} "
          f"({100.0 * len(hadm_without_icu) / n_hadm_total:0.2f}%)")


# ---------------------------------------------------------------------
# Q5 – Measurements per ICU stay + label / category + value vs valuenum vs valueuom
# ---------------------------------------------------------------------
def q5_measurements():
    """
    5. For 432,997,491 measurements:
       - Does every ICU stay (stay_id) have ≥1 measurement?
       - Unique measurements_label, measurements_category (and 5 examples)
       - Show a small sample to illustrate value / valuenum / valueuom.
    """

    print_section("Q5: Measurements per ICU stay and value vs valuenum vs valueuom")

    icu_path  = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    meas_path = os.path.join(ICU_PROC_DIR, "measurements_clean.parquet")

    icu = pd.read_parquet(icu_path, columns=["subject_id", "hadm_id", "stay_id"])
    meas = pd.read_parquet(
        meas_path,
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "measurements_label",
            "measurements_category",
            "measurements_value",
            "measurements_valuenum",
            "measurements_valueuom",
        ],
    )

    n_stay_total  = icu["stay_id"].nunique()
    n_meas_rows   = len(meas)

    stays_with_meas = meas["stay_id"].unique()
    stays_all       = icu["stay_id"].unique()

    stays_without_meas = np.setdiff1d(stays_all, stays_with_meas)
    n_with_meas    = stays_with_meas.shape[0]
    n_without_meas = stays_without_meas.shape[0]

    print(f"Total unique stay_id (ICU stays)          : {n_stay_total}")
    print(f"Total measurements rows                   : {n_meas_rows}")
    print(f"Average measurements per ICU stay         : {n_meas_rows / n_stay_total:0.2f}")
    print()
    print(f"ICU stays with ≥1 measurement             : {n_with_meas} "
          f"({100.0 * n_with_meas / n_stay_total:0.2f}%)")
    print(f"ICU stays with 0 measurements             : {n_without_meas} "
          f"({100.0 * n_without_meas / n_stay_total:0.2f}%)")

    # Unique labels/categories
    print("\nUnique measurement meta:")
    print(f"- measurements_label    : {meas['measurements_label'].nunique()}")
    print(f"- measurements_category : {meas['measurements_category'].nunique()}")

    print("\nExample measurements_label (5):")
    print(meas["measurements_label"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample measurements_category (5):")
    print(meas["measurements_category"].dropna().drop_duplicates().head(5).tolist())

    # Illustrate difference between value / valuenum / valueuom
    print("\nSample rows showing value vs valuenum vs valueuom:")
    sample = (
        meas[["measurements_label",
              "measurements_value",
              "measurements_valuenum",
              "measurements_valueuom"]]
        .dropna(subset=["measurements_value"])
        .sample(n=10, random_state=0)
    )
    print(sample.to_string(index=False))

    print(
        "\nInterpretation:\n"
        "- measurements_value    = original text entry (e.g., '120', '>200', 'Normal')\n"
        "- measurements_valuenum = parsed numeric value when possible (e.g., 120.0)\n"
        "- measurements_valueuom = unit, e.g. 'mmHg', 'bpm', '°C'\n"
        "When value is non-numeric text or ranges, valuenum is often NaN."
    )


# ---------------------------------------------------------------------
# Q6 – Medications per ICU stay + label / category
# ---------------------------------------------------------------------
def q6_medications():
    """
    6. For 10,953,713 medications:
       - Does every ICU stay have ≥1 medication row?
       - Unique medications_label, medications_category (and 5 examples)
    """

    print_section("Q6: Medications per ICU stay and medication meta")

    icu_path  = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    meds_path = os.path.join(ICU_PROC_DIR, "medications_clean.parquet")

    icu  = pd.read_parquet(icu_path,  columns=["subject_id", "hadm_id", "stay_id"])
    meds = pd.read_parquet(
        meds_path,
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "medications_label",
            "medications_category",
        ],
    )

    n_stay_total = icu["stay_id"].nunique()
    n_meds_rows  = len(meds)

    stays_with_meds = meds["stay_id"].unique()
    stays_all       = icu["stay_id"].unique()

    stays_without_meds = np.setdiff1d(stays_all, stays_with_meds)
    n_with_meds    = stays_with_meds.shape[0]
    n_without_meds = stays_without_meds.shape[0]

    print(f"Total unique stay_id (ICU stays)          : {n_stay_total}")
    print(f"Total medications rows                     : {n_meds_rows}")
    print(f"Average medication rows per ICU stay       : {n_meds_rows / n_stay_total:0.2f}")
    print()
    print(f"ICU stays with ≥1 medication row           : {n_with_meds} "
          f"({100.0 * n_with_meds / n_stay_total:0.2f}%)")
    print(f"ICU stays with 0 medication rows           : {n_without_meds} "
          f"({100.0 * n_without_meds / n_stay_total:0.2f}%)")

    print("\nUnique medication meta:")
    print(f"- medications_label    : {meds['medications_label'].nunique()}")
    print(f"- medications_category : {meds['medications_category'].nunique()}")

    print("\nExample medications_label (5):")
    print(meds["medications_label"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample medications_category (5):")
    print(meds["medications_category"].dropna().drop_duplicates().head(5).tolist())


# ---------------------------------------------------------------------
# Q7 – Outputevents per ICU stay + label / category
# ---------------------------------------------------------------------
def q7_outputevents():
    """
    7. For 5,359,395 outputevents:
       - Does every ICU stay have ≥1 outputevent row?
       - Unique outputevents_label, outputevents_category (and 5 examples)
    """

    print_section("Q7: Outputevents per ICU stay and output meta")

    icu_path   = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    out_path   = os.path.join(ICU_PROC_DIR, "outputevents_clean.parquet")

    icu  = pd.read_parquet(icu_path, columns=["subject_id", "hadm_id", "stay_id"])
    out  = pd.read_parquet(
        out_path,
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "outputevents_label",
            "outputevents_category",
        ],
    )

    n_stay_total   = icu["stay_id"].nunique()
    n_out_rows     = len(out)

    stays_with_out = out["stay_id"].unique()
    stays_all      = icu["stay_id"].unique()

    stays_without_out = np.setdiff1d(stays_all, stays_with_out)
    n_with_out    = stays_with_out.shape[0]
    n_without_out = stays_without_out.shape[0]

    print(f"Total unique stay_id (ICU stays)          : {n_stay_total}")
    print(f"Total outputevents rows                    : {n_out_rows}")
    print(f"Average outputevents per ICU stay          : {n_out_rows / n_stay_total:0.2f}")
    print()
    print(f"ICU stays with ≥1 outputevent              : {n_with_out} "
          f"({100.0 * n_with_out / n_stay_total:0.2f}%)")
    print(f"ICU stays with 0 outputevents              : {n_without_out} "
          f"({100.0 * n_without_out / n_stay_total:0.2f}%)")

    print("\nUnique outputevents meta:")
    print(f"- outputevents_label    : {out['outputevents_label'].nunique()}")
    print(f"- outputevents_category : {out['outputevents_category'].nunique()}")

    print("\nExample outputevents_label (5):")
    print(out["outputevents_label"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample outputevents_category (5):")
    print(out["outputevents_category"].dropna().drop_duplicates().head(5).tolist())


# ---------------------------------------------------------------------
# Q8 – ICU procedureevents per ICU stay + label / category
# ---------------------------------------------------------------------
def q8_procedureevents():
    """
    8. For 808,706 procedureevents:
       - Does every ICU stay have ≥1 procedureevent?
       - Unique procedureevents_label, procedureevents_category (and 5 examples)
    """

    print_section("Q8: ICU procedureevents per ICU stay and procedureevent meta")

    icu_path  = os.path.join(ICU_PROC_DIR, "icustays_clean.parquet")
    pe_path   = os.path.join(ICU_PROC_DIR, "procedureevents_clean.parquet")

    icu = pd.read_parquet(icu_path, columns=["subject_id", "hadm_id", "stay_id"])
    pe  = pd.read_parquet(
        pe_path,
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "procedureevents_label",
            "procedureevents_category",
        ],
    )

    n_stay_total = icu["stay_id"].nunique()
    n_pe_rows    = len(pe)

    stays_with_pe = pe["stay_id"].unique()
    stays_all     = icu["stay_id"].unique()

    stays_without_pe = np.setdiff1d(stays_all, stays_with_pe)
    n_with_pe    = stays_with_pe.shape[0]
    n_without_pe = stays_without_pe.shape[0]

    print(f"Total unique stay_id (ICU stays)          : {n_stay_total}")
    print(f"Total ICU procedureevents rows             : {n_pe_rows}")
    print(f"Average ICU procedureevents per ICU stay   : {n_pe_rows / n_stay_total:0.2f}")
    print()
    print(f"ICU stays with ≥1 ICU procedureevent       : {n_with_pe} "
          f"({100.0 * n_with_pe / n_stay_total:0.2f}%)")
    print(f"ICU stays with 0 ICU procedureevents       : {n_without_pe} "
          f"({100.0 * n_without_pe / n_stay_total:0.2f}%)")

    print("\nUnique ICU procedureevents meta:")
    print(f"- procedureevents_label    : {pe['procedureevents_label'].nunique()}")
    print(f"- procedureevents_category : {pe['procedureevents_category'].nunique()}")

    print("\nExample procedureevents_label (5):")
    print(pe["procedureevents_label"].dropna().drop_duplicates().head(5).tolist())

    print("\nExample procedureevents_category (5):")
    print(pe["procedureevents_category"].dropna().drop_duplicates().head(5).tolist())


# ---------------------------------------------------------------------
# Q9 – Relationship between HOSP procedures and ICU procedureevents
# ---------------------------------------------------------------------
def q9_compare_procedures():
    """
    9. Compare HOSP procedures vs ICU procedureevents:

       - HOSP: procedures_clean (ICD-coded, admission-level)
       - ICU:  procedureevents_clean (time-stamped ICU events)

       This code:
       - Checks how many hadm_id appear in both tables
       - Shows example hadm_id where both exist and prints sample names
    """

    print_section("Q9: Relationship between HOSP procedures and ICU procedureevents")

    hosp_proc_path = os.path.join(HOSP_PROC_DIR, "procedures_clean.parquet")
    icu_pe_path    = os.path.join(ICU_PROC_DIR, "procedureevents_clean.parquet")

    # Only load columns we need
    hosp_proc = pd.read_parquet(
        hosp_proc_path,
        columns=["subject_id", "hadm_id", "proc_long_title"],
    )
    icu_pe = pd.read_parquet(
        icu_pe_path,
        columns=["subject_id", "hadm_id", "stay_id", "procedureevents_label"],
    )

    hadm_hosp = set(hosp_proc["hadm_id"].unique())
    hadm_icu  = set(icu_pe["hadm_id"].unique())

    hadm_both = hadm_hosp & hadm_icu

    print(f"Unique hadm_id in HOSP procedures         : {len(hadm_hosp)}")
    print(f"Unique hadm_id in ICU procedureevents     : {len(hadm_icu)}")
    print(f"hadm_id present in BOTH tables            : {len(hadm_both)}")

    if len(hadm_both) == 0:
        print("\nNo overlapping hadm_id between HOSP procedures and ICU procedureevents.")
        return

    # Show a few example admissions
    example_hadm = list(hadm_both)[:5]
    print("\nExample hadm_id present in both tables:", example_hadm)

    for h in example_hadm:
        print("\n--- hadm_id:", h, "---")

        hosp_rows = hosp_proc[hosp_proc["hadm_id"] == h]
        icu_rows  = icu_pe[icu_pe["hadm_id"] == h]

        print("HOSP procedures (proc_long_title):")
        print(
            hosp_rows["proc_long_title"]
            .dropna()
            .drop_duplicates()
            .head(10)
            .to_string(index=False)
        )

        print("\nICU procedureevents (procedureevents_label):")
        print(
            icu_rows["procedureevents_label"]
            .dropna()
            .drop_duplicates()
            .head(10)
            .to_string(index=False)
        )

    print(
        "\nInterpretation:\n"
        "- HOSP procedures (procedures_clean) are ICD-coded procedures recorded "
        "at the admission level (e.g., 'CORONARY ANGIOGRAPHY').\n"
        "- ICU procedureevents (procedureevents_clean) are granular bedside ICU "
        "procedures tied to stay_id and exact times (e.g., 'Mechanical ventilation', "
        "'Central line insertion').\n"
        "- The overlap by hadm_id tells you which admissions have both types of "
        "information, but the procedure names will often differ in granularity and wording."
    )


# ---------------------------------------------------------------------
# Q10 – Discharge notes coverage
# ---------------------------------------------------------------------
def q10_discharge_notes():
    """
    10. Discharge notes coverage:

        - How many admissions have a discharge note?
        - Split: exactly 1 vs >1 discharge notes per hadm_id.
    """

    print_section("Q10: Discharge notes coverage (including 1 vs >1 split)")

    adm_path   = os.path.join(HOSP_PROC_DIR,  "patients_admissions_clean.parquet")
    notes_path = os.path.join(NOTES_PROC_DIR, "discharge_clean.parquet")

    adm   = pd.read_parquet(adm_path,   columns=["subject_id", "hadm_id"])
    notes = pd.read_parquet(notes_path, columns=["subject_id", "hadm_id", "note_id"])

    # Basic coverage
    n_hadm_total    = adm["hadm_id"].nunique()
    hadm_with_notes = notes["hadm_id"].unique()
    n_hadm_with_notes = len(hadm_with_notes)

    coverage = n_hadm_with_notes / n_hadm_total

    print(f"Total unique hadm_id (hospital admissions): {n_hadm_total}")
    print(f"Admissions with ≥1 discharge note          : {n_hadm_with_notes}")
    print(f"Coverage                                   : {coverage*100:0.2f}%")

    hadm_without_notes = np.setdiff1d(adm["hadm_id"].unique(), hadm_with_notes)
    print(f"Admissions with 0 discharge notes          : {len(hadm_without_notes)} "
          f"({100.0 * len(hadm_without_notes) / n_hadm_total:0.2f}%)")

    # --- New part: split by number of discharge notes per hadm_id ---
    # Count how many distinct notes each hadm_id has
    notes_per_hadm = (
        notes.groupby("hadm_id")["note_id"]
        .nunique()
        .rename("n_discharge_notes")
    )

    n_exactly_1 = (notes_per_hadm == 1).sum()
    n_more_than_1 = (notes_per_hadm > 1).sum()

    print("\nBreakdown among admissions WITH ≥1 discharge note:")
    print(f"- Admissions with exactly 1 discharge note : {n_exactly_1} "
          f"({100.0 * n_exactly_1 / n_hadm_with_notes:0.2f}% of those with notes)")
    print(f"- Admissions with >1 discharge note        : {n_more_than_1} "
          f"({100.0 * n_more_than_1 / n_hadm_with_notes:0.2f}% of those with notes)")
    print(f"- Max number of discharge notes for any admission: {notes_per_hadm.max()}")

    if n_more_than_1 > 0:
        # Show a few example hadm_id with multiple discharge notes
        multi = notes_per_hadm[notes_per_hadm > 1].sort_values(ascending=False)
        example_multi = multi.head(5)
        print("\nExample hadm_id with multiple discharge notes:")
        print(example_multi.to_string())

        # Optionally, show the note_ids for one example hadm_id
        example_hadm = example_multi.index[0]
        example_notes = notes[notes["hadm_id"] == example_hadm][["hadm_id", "note_id"]]
        print(f"\nnote_id values for example hadm_id={example_hadm}:")
        print(example_notes.drop_duplicates().to_string(index=False))

    print(
        "\nInterpretation (why ~60% and not 100%?):\n"
        "- MIMIC includes only notes that passed de-identification and meet "
        "specific inclusion criteria; not every admission's documentation made it into "
        "the released NOTEEVENTS table.\n"
        "- In your 'discharge_clean' table, you are filtering to notes with "
        "category='Discharge summary'. Some admissions may have other note types "
        "(e.g., progress notes) but no discharge summary entry.\n"
        "- There may also be short stays, observation-only admissions, or technical "
        "gaps where a discharge summary simply wasn't captured or linked.\n"
        "So it's expected that only a subset of admissions have a clean, linked "
        "discharge summary in this processed dataset."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # You can comment out any heavy sections while experimenting.
    q1_diagnoses()
    q2_hosp_procedures()
    q3_lab_tests()
    q4_icu_vs_hosp()
    q5_measurements()
    q6_medications()
    q7_outputevents()
    q8_procedureevents()
    q9_compare_procedures()
    q10_discharge_notes()


if __name__ == "__main__":
    main()