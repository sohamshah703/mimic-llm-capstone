import os

# Raw data locations
MIMIC_IV_DIR = "/scratch/soham_shah/mimiciv/mimic-iv-3.1"
MIMIC_NOTES_DIR = "/scratch/soham_shah/mimiciv_note/mimiciv-clinical-notes/note"

# Processed data root
PROC_DIR = "/scratch/soham_shah/mimic_proc_data"

# Subfolders
HOSP_DIR = os.path.join(MIMIC_IV_DIR, "hosp")
ICU_DIR = os.path.join(MIMIC_IV_DIR, "icu")

HOSP_PROC_DIR = os.path.join(PROC_DIR, "hosp")
ICU_PROC_DIR = os.path.join(PROC_DIR, "icu")
NOTES_PROC_DIR = os.path.join(PROC_DIR, "notes")

# Make sure processed folders exist whenever this is imported
os.makedirs(HOSP_PROC_DIR, exist_ok=True)
os.makedirs(ICU_PROC_DIR, exist_ok=True)
os.makedirs(NOTES_PROC_DIR, exist_ok=True)

# ---- Cohort-processed data locations ----

PROC_COHORT_DIR = "/scratch/soham_shah/mimic_proc_data_cohort"

COHORT_META_DIR = os.path.join(PROC_COHORT_DIR, "meta")
HOSP_PROC_COHORT_DIR = os.path.join(PROC_COHORT_DIR, "hosp")
ICU_PROC_COHORT_DIR = os.path.join(PROC_COHORT_DIR, "icu")
NOTES_PROC_COHORT_DIR = os.path.join(PROC_COHORT_DIR, "notes")

os.makedirs(COHORT_META_DIR, exist_ok=True)
os.makedirs(HOSP_PROC_COHORT_DIR, exist_ok=True)
os.makedirs(ICU_PROC_COHORT_DIR, exist_ok=True)
os.makedirs(NOTES_PROC_COHORT_DIR, exist_ok=True)
