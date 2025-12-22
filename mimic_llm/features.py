import os
import sys
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# --- wiring to import paths.py from project root ---

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import (  # type: ignore
    COHORT_META_DIR,
    ICU_PROC_COHORT_DIR,
    HOSP_PROC_COHORT_DIR,
    NOTES_PROC_COHORT_DIR,
)


# -------------------------------------------------------------------
# SMALL HELPER UTILITIES
# -------------------------------------------------------------------


def _first_non_null(series: pd.Series) -> Any:
    """Return the first non-null value in a Series, or None."""
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _safe_get_single_row(df: pd.DataFrame, where: str) -> pd.Series:
    """
    Return the first row of df if not empty, else an empty Series.

    `where` is only used for error/debug messages.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype="object")
    return df.iloc[0]


def _calculate_trend(df: pd.DataFrame, time_col: str, val_col: str) -> str:
    """
    Determines if values are Rising, Falling, or Stable over time.
    Uses simple linear regression slope.
    """
    df = df.dropna(subset=[time_col, val_col]).sort_values(time_col)
    if len(df) < 3:
        return "Insufficient data"

    # Convert time to numeric (seconds/hours) for regression
    # (timestamp - min_timestamp) / 1 hour
    x = (df[time_col] - df[time_col].min()).dt.total_seconds() / 3600.0
    y = df[val_col].values

    # Fit line: y = mx + c
    # slope (m) represents change per hour
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except:
        return "Stable"

    # Calculate relative change over the whole period
    total_time_hours = x.max()
    if total_time_hours == 0:
        return "Stable"
    
    total_change = slope * total_time_hours
    start_val = (slope * x.min()) + intercept
    
    # Avoid division by zero
    if start_val == 0: 
        baseline = np.mean(y)
    else:
        baseline = start_val

    pct_change = (total_change / (baseline + 1e-9)) * 100

    # Define thresholds (e.g., >10% change is significant)
    if pct_change > 10:
        return "Rising"
    elif pct_change < -10:
        return "Falling"
    else:
        return "Stable"

# -------------------------------------------------------------------
# CORE LOADER
# -------------------------------------------------------------------

def load_all_tables_for_stay(stay_id: int) -> Dict[str, Any]:
    """
    Load all relevant cohort-filtered tables for a single stay_id.

    Returns a dictionary with:
    - 'stay_id', 'hadm_id', 'subject_id'
    - 'cohort_row' : the row from cohort_icu_250
    - 'icu'  : dict of ICU tables for that stay
    - 'hosp' : dict of hosp tables for that stay/hadm
    - 'discharge_text' : full discharge summary for that hadm_id
    """

    # 1. Load cohort and find the row for this stay_id
    cohort_path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    cohort = pd.read_parquet(cohort_path)

    if stay_id not in set(cohort["stay_id"].unique()):
        raise ValueError(f"stay_id {stay_id} not found in cohort_icu_250")

    cohort_row = cohort[cohort["stay_id"] == stay_id].iloc[0]
    hadm_id = int(cohort_row["hadm_id"])
    subject_id = int(cohort_row["subject_id"])

    # 2. Load ICU tables (already cohort-filtered) and slice by stay_id
    def load_icu_table(name: str) -> pd.DataFrame:
        path = os.path.join(ICU_PROC_COHORT_DIR, name)
        return pd.read_parquet(path)

    icustays = load_icu_table("icustays_clean_icu_250.parquet")
    measurements = load_icu_table("measurements_clean_icu_250.parquet")
    medications = load_icu_table("medications_clean_icu_250.parquet")
    outputevents = load_icu_table("outputevents_clean_icu_250.parquet")
    proc_icu = load_icu_table("procedureevents_clean_icu_250.parquet")

    icustays_this = icustays[icustays["stay_id"] == stay_id].copy()
    measurements_this = measurements[measurements["stay_id"] == stay_id].copy()
    medications_this = medications[medications["stay_id"] == stay_id].copy()
    outputevents_this = outputevents[outputevents["stay_id"] == stay_id].copy()
    proc_icu_this = proc_icu[proc_icu["stay_id"] == stay_id].copy()

    icu_tables = {
        "icustays": icustays_this,
        "measurements": measurements_this,
        "medications": medications_this,
        "outputevents": outputevents_this,
        "procedureevents": proc_icu_this,
    }

    # 3. Load hosp tables (cohort-filtered) and slice by hadm_id / stay_id
    def load_hosp_table(name: str) -> pd.DataFrame:
        path = os.path.join(HOSP_PROC_COHORT_DIR, name)
        return pd.read_parquet(path)

    patadm = load_hosp_table("patients_admissions_clean_icu_250.parquet")
    diagnoses = load_hosp_table("diagnoses_clean_icu_250.parquet")
    procedures = load_hosp_table("procedures_clean_icu_250.parquet")
    labs = load_hosp_table("lab_tests_clean_icu_250.parquet")

    patadm_this = patadm[patadm["hadm_id"] == hadm_id].copy()
    dx_this = diagnoses[diagnoses["hadm_id"] == hadm_id].copy()

    # Procedures & labs ICU-window tables include stay_id; if not, fall back to hadm_id
    if "stay_id" in procedures.columns:
        procs_this = procedures[procedures["stay_id"] == stay_id].copy()
    else:
        procs_this = procedures[procedures["hadm_id"] == hadm_id].copy()

    if "stay_id" in labs.columns:
        labs_this = labs[labs["stay_id"] == stay_id].copy()
    else:
        labs_this = labs[labs["hadm_id"] == hadm_id].copy()

    hosp_tables = {
        "patients_admissions": patadm_this,
        "diagnoses": dx_this,
        "procedures": procs_this,
        "labs": labs_this,
    }

    # 4. Load discharge summary (cohort-filtered)
    discharge_path = os.path.join(
        NOTES_PROC_COHORT_DIR, "discharge_clean_icu_250.parquet"
    )
    discharge = pd.read_parquet(discharge_path)
    disc_this = discharge[discharge["hadm_id"] == hadm_id].copy()

    if len(disc_this) == 0:
        discharge_text = ""
    else:
        # We enforce exactly 1 discharge note per hadm_id in cohort building,
        # but still just take the first row defensively.
        discharge_text = str(disc_this.iloc[0].get("text", ""))

    return {
        "stay_id": stay_id,
        "hadm_id": hadm_id,
        "subject_id": subject_id,
        "cohort_row": cohort_row,
        "icu": icu_tables,
        "hosp": hosp_tables,
        "discharge_text": discharge_text,
    }


# -------------------------------------------------------------------
# VIEW: DEMOGRAPHICS / ADMISSION
# -------------------------------------------------------------------


def build_view_demographics(stay_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a 'demographics/admission' view.

    Focuses on:
    - anchor_age
    - gender
    - admission_type
    - admission_location
    - discharge_location
    - admittime
    - dischtime
    - hospital_expire_flag
    - deathdate
    """

    patadm = stay_data["hosp"]["patients_admissions"].copy()
    row = _safe_get_single_row(patadm, "patients_admissions")

    def get_safe(col: str, default=None):
        return row[col] if col in row.index else default

    demographics = {
        "anchor_age": get_safe("anchor_age"),
        "gender": get_safe("gender"),
        "admission_type": get_safe("admission_type"),
        "admission_location": get_safe("admission_location"),
        "discharge_location": get_safe("discharge_location"),
        "admittime": get_safe("admittime"),
        "dischtime": get_safe("dischtime"),
        "hospital_expire_flag": get_safe("hospital_expire_flag"),
        "deathdate": get_safe("deathdate"),
    }

    return {"demographics": demographics}


# -------------------------------------------------------------------
# VIEW: DIAGNOSES (HOSPITAL)
# -------------------------------------------------------------------


def build_view_diagnoses(stay_data: Dict[str, Any], max_diagnoses: int = 15) -> Dict[str, Any]:
    """
    Build the 'diagnoses-only' view for a stay.

    Uses:
    - diagnoses_clean_icu_250 (HOSP table, filtered by hadm_id)

    We only care about:
    - dx_long_title (human-readable diagnosis)
    - dx_seq_num (for ordering)

    Returns:
        {
            "diagnoses": [
                {"sequence": 1, "title": "..."},
                ...
            ]
        }
    """

    dx = stay_data["hosp"]["diagnoses"].copy()

    if "dx_long_title" not in dx.columns:
        return {"diagnoses": []}

    # Sort by sequence number if available
    if "dx_seq_num" in dx.columns:
        dx_sorted = dx.sort_values(["hadm_id", "dx_seq_num"])
    else:
        dx_sorted = dx.copy()

    diagnoses_list: List[Dict[str, Any]] = []
    for _, r in dx_sorted.iterrows():
        diagnoses_list.append(
            {
                "sequence": r.get("dx_seq_num"),
                "title": r.get("dx_long_title"),
            }
        )

    diagnoses_list = diagnoses_list[:max_diagnoses]
    return {"diagnoses": diagnoses_list}


# -------------------------------------------------------------------
# VIEW: HOSPITAL PROCEDURES (ICU-window)
# -------------------------------------------------------------------


def build_view_hosp_procedures(
    stay_data: Dict[str, Any], max_procs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build the 'HOSP procedures during ICU window' view for a stay.

    Uses:
    - procedures_clean_icu_250, filtered by stay_id (ICU-window pre-filtered)

    We care about:
    - proc_seq_num (for ordering if present)
    - proc_long_title (human-readable procedure)
    - procedure_chartdatetime (approximate time, if present)
    """

    procs = stay_data["hosp"]["procedures"].copy()
    if procs.empty:
        return {"procedures_hosp": []}

    # Identify key columns
    title_col = "proc_long_title" if "proc_long_title" in procs.columns else None
    if title_col is None:
        # fall back to any 'long_title' or 'label' column
        for c in procs.columns:
            if "long_title" in c or "label" in c:
                title_col = c
                break

    time_col = None
    for c in procs.columns:
        if "charttime" in c or "datetime" in c or c.endswith("_date") or c.endswith("_time"):
            time_col = c
            break

    seq_col = "proc_seq_num" if "proc_seq_num" in procs.columns else None

    # Sort by sequence or time
    if seq_col is not None:
        procs = procs.sort_values([seq_col])
    elif time_col is not None:
        procs = procs.sort_values([time_col])

    procedures_list: List[Dict[str, Any]] = []
    for _, r in procs.iterrows():
        procedures_list.append(
            {
                "sequence": r.get(seq_col) if seq_col is not None else None,
                "title": r.get(title_col) if title_col is not None else None,
                "time": r.get(time_col) if time_col is not None else None,
            }
        )

    if max_procs is not None:
        procedures_list = procedures_list[:max_procs]

    return {"procedures_hosp": procedures_list}


# -------------------------------------------------------------------
# VIEW: ICU PROCEDUREEVENTS (bedside procedures)
# -------------------------------------------------------------------


def build_view_icu_procedures(
    stay_data: Dict[str, Any], max_events: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build the ICU 'procedureevents' view for a stay.

    Uses:
    - procedureevents_clean_icu_250 (ICU table, filtered by stay_id)

    We focus on:
    - procedureevents_label
    - procedureevents_category
    - procedureevents_location
    - procedureevents_value, procedureevents_valueuom
    - procedureevents_start_date, procedureevents_end_date

    Returns:
        {
            "procedureevents": [
                {
                    "label": ...,
                    "category": ...,
                    "location": ...,
                    "value": ...,
                    "valueuom": ...,
                    "start": ...,
                    "end": ...,
                },
                ...
            ]
        }
    """

    proc_icu = stay_data["icu"]["procedureevents"].copy()
    if proc_icu.empty:
        return {"procedureevents": []}

    label_col = "procedureevents_label" if "procedureevents_label" in proc_icu.columns else None
    cat_col = "procedureevents_category" if "procedureevents_category" in proc_icu.columns else None
    loc_col = "procedureevents_location" if "procedureevents_location" in proc_icu.columns else None
    val_col = "procedureevents_value" if "procedureevents_value" in proc_icu.columns else None
    val_uom_col = (
        "procedureevents_valueuom" if "procedureevents_valueuom" in proc_icu.columns else None
    )

    start_col = None
    for c in proc_icu.columns:
        if "start" in c and ("date" in c or "time" in c):
            start_col = c
            break

    end_col = None
    for c in proc_icu.columns:
        if "end" in c and ("date" in c or "time" in c):
            end_col = c
            break

    # Sort by start time if available
    if start_col is not None:
        proc_icu = proc_icu.sort_values([start_col])

    events: List[Dict[str, Any]] = []
    for _, r in proc_icu.iterrows():
        events.append(
            {
                "label": r.get(label_col) if label_col is not None else None,
                "category": r.get(cat_col) if cat_col is not None else None,
                "location": r.get(loc_col) if loc_col is not None else None,
                "value": r.get(val_col) if val_col is not None else None,
                "valueuom": r.get(val_uom_col) if val_uom_col is not None else None,
                "start": r.get(start_col) if start_col is not None else None,
                "end": r.get(end_col) if end_col is not None else None,
            }
        )

    if max_events is not None:
        events = events[:max_events]

    return {"procedureevents_summary": events}  


# -------------------------------------------------------------------
# VIEW: Procedure events
# -------------------------------------------------------------------

def build_view_procedureevents(
    stay_data: Dict[str, Any], max_events: Optional[int] = None
) -> Dict[str, Any]:
    """
    Thin alias for ICU procedureevents, for clarity.

    This makes it clear that these are the ICU `procedureevents` table
    (not HOSP procedures). It simply forwards to `build_view_icu_procedures`.
    """
    return build_view_icu_procedures(stay_data, max_events=max_events)


# -------------------------------------------------------------------
# VIEW: LABS (ICU-window)
# -------------------------------------------------------------------

def build_view_labs(stay_data: Dict[str, Any], max_labs: int = 10) -> Dict[str, Any]:
    """
    Build the 'labs-only' view for a stay.
    Now includes Trend calculation and cleaner Unit extraction.
    """
    labs = stay_data["hosp"]["labs"].copy()
    if labs.empty:
        return {"labs_summary": []}

    # Identify key columns
    label_col = None
    for c in labs.columns:
        if c == "lab_tests_label" or (c.endswith("label") and "lab_tests_" in c):
            label_col = c
            break

    val_col = None
    for c in labs.columns:
        if "valuenum" in c:
            val_col = c
            break

    # Identify unit column
    unit_col = None
    for c in labs.columns:
        if "valueuom" in c:
            unit_col = c
            break

    warning_col = None
    for c in labs.columns:
        if "warning" in c:
            warning_col = c
            break

    fluid_col = "lab_tests_fluid" if "lab_tests_fluid" in labs.columns else None
    cat_col = "lab_tests_category" if "lab_tests_category" in labs.columns else None

    # Identify time column for trends
    time_col = None
    for c in labs.columns:
        if "charttime" in c or "datetime" in c or c.endswith("_time") or c.endswith("_date"):
            time_col = c
            break

    if label_col is None or val_col is None:
        return {"labs_summary": []}

    labs[val_col] = pd.to_numeric(labs[val_col], errors="coerce")

    if warning_col is not None:
        labs[warning_col] = pd.to_numeric(labs[warning_col], errors="coerce").fillna(0)

    grouped = labs.groupby(label_col)
    labs_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        vals = group[val_col].dropna()
        if len(vals) == 0:
            continue

        count = len(group)
        min_val = vals.min()
        max_val = vals.max()
        median_val = vals.median()

        if warning_col is not None:
            abnormal_count = int((group[warning_col] == 1).sum())
        else:
            abnormal_count = 0

        # 1. Extract Unit (Mode)
        unit = None
        if unit_col is not None and unit_col in group.columns:
            uoms = group[unit_col].dropna()
            if not uoms.empty:
                unit = uoms.mode().iloc[0]

        # 2. Calculate Trend
        trend = "Unknown"
        if time_col is not None and time_col in group.columns:
            trend = _calculate_trend(group, time_col, val_col)

        fluid = (
            _first_non_null(group[fluid_col]) if fluid_col is not None and fluid_col in group.columns else None
        )
        category = (
            _first_non_null(group[cat_col]) if cat_col is not None and cat_col in group.columns else None
        )

        labs_summary.append(
            {
                "lab_name": str(label),
                "fluid": None if pd.isna(fluid) else fluid,
                "category": None if pd.isna(category) else category,
                "count": int(count),
                "abnormal_count": abnormal_count,
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val),
                "unit": unit,   # <--- Added
                "trend": trend, # <--- Added
            }
        )

    # Sort by abnormal_count (desc), then by count
    labs_summary = sorted(
        labs_summary,
        key=lambda x: (x["abnormal_count"], x["count"]),
        reverse=True,
    )

    return {"labs_summary": labs_summary[:max_labs]}


# -------------------------------------------------------------------
# VIEW: MEDICATIONS (ICU meds)
# -------------------------------------------------------------------

def build_view_meds(stay_data: Dict[str, Any], max_meds: int = 10) -> Dict[str, Any]:
    """
    Build the 'meds-only' view for a stay.
    UPDATED: Performs 'Safe Summation' by only summing amounts that match the dominant unit.
    """
    meds = stay_data["icu"]["medications"].copy()
    if meds.empty:
        return {"meds_summary": []}

    # Identify label col
    label_col = None
    for c in meds.columns:
        if c == "medications_label" or (c.endswith("label") and "medications_" in c):
            label_col = c
            break
    if label_col is None:
        for c in meds.columns:
            if "label" in c:
                label_col = c
                break

    if label_col is None:
        return {"meds_summary": []}

    cat_col = "medications_category" if "medications_category" in meds.columns else None

    # Identify amount & unit columns
    amount_col = None
    for c in meds.columns:
        if "amount" in c and "original" not in c and "uom" not in c:
            amount_col = c
            break
    
    amount_uom_col = None
    for c in meds.columns:
        if "amountuom" in c and "original" not in c:
            amount_uom_col = c
            break

    # Identify time columns
    start_col = None
    for c in meds.columns:
        if "start" in c and ("date" in c or "time" in c):
            start_col = c
            break
    end_col = None
    for c in meds.columns:
        if "end" in c and ("date" in c or "time" in c):
            end_col = c
            break

    grouped = meds.groupby(label_col)
    meds_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        count = len(group)

        # Category (take most frequent)
        category = None
        if cat_col is not None and cat_col in group.columns:
            non_null = group[cat_col].dropna()
            category = non_null.mode().iloc[0] if not non_null.empty else None

        # --- SAFE SUMMATION LOGIC ---
        total_amount = None
        unit = None

        # 1. Determine the Dominant Unit (Mode)
        if amount_uom_col is not None and amount_uom_col in group.columns:
            uoms = group[amount_uom_col].dropna()
            if not uoms.empty:
                unit = uoms.mode().iloc[0]
        
        # 2. Sum ONLY rows that match the dominant unit
        if amount_col is not None and amount_col in group.columns:
            # If we found a unit, filter by it
            if unit is not None and amount_uom_col in group.columns:
                mask = group[amount_uom_col] == unit
                vals = pd.to_numeric(group.loc[mask, amount_col], errors="coerce").dropna()
            else:
                # Fallback: if no unit info exists, sum everything (legacy behavior)
                vals = pd.to_numeric(group[amount_col], errors="coerce").dropna()
            
            if not vals.empty:
                total_amount = float(vals.sum())

        # Start/end times
        start_min = None
        end_max = None
        if start_col is not None and start_col in group.columns:
            start_times = pd.to_datetime(group[start_col], errors="coerce")
            if not start_times.dropna().empty:
                start_min = start_times.min()
        if end_col is not None and end_col in group.columns:
            end_times = pd.to_datetime(group[end_col], errors="coerce")
            if not end_times.dropna().empty:
                end_max = end_times.max()

        meds_summary.append(
            {
                "med_name": str(label),
                "category": None if pd.isna(category) else category,
                "num_orders": int(count),
                "total_amount": total_amount,
                "unit": unit,
                "first_start": start_min,
                "last_end": end_max,
            }
        )

    # Sort: more frequently used meds first
    meds_summary = sorted(
        meds_summary,
        key=lambda x: x["num_orders"],
        reverse=True,
    )

    return {"meds_summary": meds_summary[:max_meds]}


"""
def build_view_meds(stay_data: Dict[str, Any], max_meds: int = 25) -> Dict[str, Any]:
    
    Build the 'meds-only' view for a stay.

    Uses:
    - medications_clean_icu_250 (ICU meds table, filtered by stay_id)

    Aggregates by medication label/category:
    - num_orders
    - total_amount
    - first_start
    - last_end
    

    meds = stay_data["icu"]["medications"].copy()
    if meds.empty:
        return {"meds_summary": []}

    # Identify label/category/time/amount cols
    label_col = None
    for c in meds.columns:
        if c == "medications_label" or (c.endswith("label") and "medications_" in c):
            label_col = c
            break
    if label_col is None:
        for c in meds.columns:
            if "label" in c:
                label_col = c
                break

    cat_col = "medications_category" if "medications_category" in meds.columns else None

    start_col = None
    for c in meds.columns:
        if "start" in c and ("date" in c or "time" in c):
            start_col = c
            break

    end_col = None
    for c in meds.columns:
        if "end" in c and ("date" in c or "time" in c):
            end_col = c
            break

    amount_col = None
    for c in meds.columns:
        if "amount" in c and "original" not in c:
            amount_col = c
            break

    if label_col is None:
        return {"meds_summary": []}

    grouped = meds.groupby(label_col)

    meds_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        count = len(group)

        # Category (take most frequent)
        category = None
        if cat_col is not None and cat_col in group.columns:
            non_null = group[cat_col].dropna()
            category = non_null.mode().iloc[0] if not non_null.empty else None

        # Amount summary if numeric
        total_amount = None
        if amount_col is not None and amount_col in group.columns:
            vals = pd.to_numeric(group[amount_col], errors="coerce").dropna()
            if not vals.empty:
                total_amount = float(vals.sum())

        # Start/end times approximate range
        start_min = None
        end_max = None
        if start_col is not None and start_col in group.columns:
            start_times = pd.to_datetime(group[start_col], errors="coerce")
            if not start_times.dropna().empty:
                start_min = start_times.min()
        if end_col is not None and end_col in group.columns:
            end_times = pd.to_datetime(group[end_col], errors="coerce")
            if not end_times.dropna().empty:
                end_max = end_times.max()

        meds_summary.append(
            {
                "med_name": str(label),
                "category": None if pd.isna(category) else category,
                "num_orders": int(count),
                "total_amount": total_amount,
                "first_start": start_min,
                "last_end": end_max,
            }
        )

    # Sort: more frequently used meds first
    meds_summary = sorted(
        meds_summary,
        key=lambda x: x["num_orders"],
        reverse=True,
    )

    # Limit length
    meds_summary = meds_summary[:max_meds]

    return {"meds_summary": meds_summary}
"""

# -------------------------------------------------------------------
# VIEW: MEASUREMENTS (ICU measurements / chartevents)
# -------------------------------------------------------------------

def build_view_measurements(
    stay_data: Dict[str, Any], max_labels: int = 10
) -> Dict[str, Any]:
    """
    Build the 'measurements-only' view for a stay.
    Now includes Unit extraction and Trend calculation.
    """
    meas = stay_data["icu"]["measurements"].copy()
    if meas.empty:
        return {"measurements_summary": []}

    # Identify label column
    label_col = None
    for c in meas.columns:
        if c == "measurements_label" or (c.endswith("label") and "measurements_" in c):
            label_col = c
            break
    if label_col is None:
        for c in meas.columns:
            if "label" in c:
                label_col = c
                break

    # Identify value column
    val_col = None
    for c in meas.columns:
        if "valuenum" in c:
            val_col = c
            break
    
    # Identify unit column
    val_uom_col = None
    for c in meas.columns:
        if "valueuom" in c:
            val_uom_col = c
            break

    # Identify time column (critical for trends)
    time_col = None
    for c in meas.columns:
        if "charttime" in c or "datetime" in c or c.endswith("_time") or c.endswith("_date"):
            time_col = c
            break

    if label_col is None or val_col is None:
        return {"measurements_summary": []}

    meas[val_col] = pd.to_numeric(meas[val_col], errors="coerce")

    # Optionally focus on the most frequent measurement labels for this stay
    label_counts = meas[label_col].value_counts()
    top_labels = label_counts.head(max_labels).index.tolist()
    meas = meas[meas[label_col].isin(top_labels)]

    grouped = meas.groupby(label_col)
    measurements_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        vals = group[val_col].dropna()
        if len(vals) == 0:
            continue

        count = len(group)
        min_val = vals.min()
        max_val = vals.max()
        median_val = vals.median()
        
        # 1. Extract Unit (Mode)
        unit = None
        if val_uom_col is not None and val_uom_col in group.columns:
            uoms = group[val_uom_col].dropna()
            if not uoms.empty:
                unit = uoms.mode().iloc[0]

        # 2. Calculate Trend
        trend = "Unknown"
        if time_col is not None and time_col in group.columns:
            # Uses the helper function you added: _calculate_trend
            trend = _calculate_trend(group, time_col, val_col)

        measurements_summary.append(
            {
                "measure_name": str(label),
                "count": int(count),
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val),
                "unit": unit,   # <--- Added
                "trend": trend, # <--- Added
            }
        )

    # Sort by count
    measurements_summary = sorted(
        measurements_summary,
        key=lambda x: x["count"],
        reverse=True,
    )

    return {"measurements_summary": measurements_summary}


"""
def build_view_measurements(
    stay_data: Dict[str, Any], max_labels: int = 20
) -> Dict[str, Any]:
    
    Build the 'measurements-only' view for a stay.

    Uses:
    - measurements_clean_icu_250 (chartevents-like ICU measurements)

    Aggregates by measurement label:
    - count
    - min / max / median of valuenum
    - optional trend (increasing / decreasing / stable)
    

    meas = stay_data["icu"]["measurements"].copy()
    if meas.empty:
        return {"measurements_summary": []}

    # Identify label & value & time columns
    label_col = None
    for c in meas.columns:
        if c == "measurements_label" or (c.endswith("label") and "measurements_" in c):
            label_col = c
            break
    if label_col is None:
        for c in meas.columns:
            if "label" in c:
                label_col = c
                break

    val_col = None
    for c in meas.columns:
        if "valuenum" in c:
            val_col = c
            break

    time_col = None
    for c in meas.columns:
        if "charttime" in c or "datetime" in c or c.endswith("_time") or c.endswith("_date"):
            time_col = c
            break

    if label_col is None or val_col is None:
        return {"measurements_summary": []}

    meas[val_col] = pd.to_numeric(meas[val_col], errors="coerce")

    # Optionally focus on the most frequent measurement labels for this stay
    label_counts = meas[label_col].value_counts()
    top_labels = label_counts.head(max_labels).index.tolist()

    meas = meas[meas[label_col].isin(top_labels)]

    grouped = meas.groupby(label_col)

    measurements_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        vals = group[val_col].dropna()
        if len(vals) == 0:
            continue

        count = len(group)
        min_val = vals.min()
        max_val = vals.max()
        median_val = vals.median()
        
         
        # Trend, if time info available
        if time_col is not None:
            trend = _infer_trend_from_times(vals, group[time_col])
        else:
            trend = None
        

        measurements_summary.append(
            {
                "measure_name": str(label),
                "count": int(count),
                "min": float(min_val),
                "max": float(max_val),
                "median": float(median_val),
                # "trend": trend,
            }
        )

    # Sort by count
    measurements_summary = sorted(
        measurements_summary,
        key=lambda x: x["count"],
        reverse=True,
    )

    return {"measurements_summary": measurements_summary}
"""

# -------------------------------------------------------------------
# VIEW: OUTPUTEVENTS (ICU outputs)
# -------------------------------------------------------------------

def build_view_outputs(
    stay_data: Dict[str, Any], max_labels: int = 15
) -> Dict[str, Any]:
    """
    Build the 'outputs-only' view for a stay.

    Uses:
    - outputevents_clean_icu_250 (ICU outputevents table)

    Aggregates by output label:
    - num_records
    - total_value (where numeric)
    - first_time / last_time
    - optional trend
    """

    out = stay_data["icu"]["outputevents"].copy()
    if out.empty:
        return {"outputs_summary": []}

    label_col = "outputevents_label" if "outputevents_label" in out.columns else None
    if label_col is None:
        for c in out.columns:
            if "label" in c:
                label_col = c
                break

    cat_col = "outputevents_category" if "outputevents_category" in out.columns else None
    val_col = "outputevents_value" if "outputevents_value" in out.columns else None
    val_uom_col = (
        "outputevents_valueuom" if "outputevents_valueuom" in out.columns else None
    )

    time_col = None
    for c in out.columns:
        if "charttime" in c or "datetime" in c or c.endswith("_time") or c.endswith("_date"):
            time_col = c
            break

    if label_col is None or val_col is None:
        return {"outputs_summary": []}

    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")

    grouped = out.groupby(label_col)

    outputs_summary: List[Dict[str, Any]] = []

    for label, group in grouped:
        vals = group[val_col].dropna()
        count = len(group)
        total_value = float(vals.sum()) if not vals.empty else None

       

        category = (
            _first_non_null(group[cat_col])
            if cat_col is not None and cat_col in group.columns
            else None
        )
        valueuom = (
            _first_non_null(group[val_uom_col])
            if val_uom_col is not None and val_uom_col in group.columns
            else None
        )

        outputs_summary.append(
            {
                "label": str(label),
                "category": None if pd.isna(category) else category,
                "num_records": int(count),
                "total_value": total_value,
                "valueuom": None if pd.isna(valueuom) else valueuom,
            }
        )

    # Sort by num_records desc
    outputs_summary = sorted(
        outputs_summary,
        key=lambda x: x["num_records"],
        reverse=True,
    )

    outputs_summary = outputs_summary[:max_labels]

    return {"outputs_summary": outputs_summary}


# -------------------------------------------------------------------
# BACKWARDS-COMPATIBLE VIEWS: DX_PROC + ADMISSION
# -------------------------------------------------------------------


def build_view_dx_proc(stay_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the combined 'diagnoses + HOSP ICU-window procedures + demographics' view.

    This is mainly for backwards compatibility with older prompt code that
    expects a single dict with keys:
        - 'demographics'
        - 'diagnoses'
        - 'procedures'

    Internally, it delegates to:
        - build_view_demographics
        - build_view_diagnoses
        - build_view_hosp_procedures

    and then exposes:
        - 'procedures' (alias of 'procedures_hosp') for compatibility
        - 'procedures_hosp' (explicit)
    """

    demo_view = build_view_demographics(stay_data)
    dx_view = build_view_diagnoses(stay_data)
    procs_view = build_view_hosp_procedures(stay_data)

    demographics = demo_view.get("demographics", {})
    diagnoses = dx_view.get("diagnoses", [])
    procedures_hosp = procs_view.get("procedures_hosp", [])

    return {
        "demographics": demographics,
        "diagnoses": diagnoses,
        # backwards-compatible key:
        "procedures": procedures_hosp,
        # more explicit key:
        "procedures_hosp": procedures_hosp,
    }


def build_view_admission(stay_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an 'admission/demographics + diagnoses' view.

    This reuses:
        - build_view_demographics
        - build_view_diagnoses
    """
    demo_view = build_view_demographics(stay_data)
    dx_view = build_view_diagnoses(stay_data)
    return {
        "demographics": demo_view.get("demographics", {}),
        "diagnoses": dx_view.get("diagnoses", []),
    }