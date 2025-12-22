"""
visuals.py

Streamlit helpers for:
- Time-series plots from numeric ICU / HOSP tables
- Small structured tables for demographics, diagnoses, and procedures

These functions operate on the `stay_data` dict returned by
`features.load_all_tables_for_stay(stay_id)`.
"""

from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# Altair is optional – if not installed, we fall back to st.line_chart.
try:
    import altair as alt  # type: ignore
except ImportError:
    alt = None


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _downsample(df: pd.DataFrame, max_points: int = 200) -> pd.DataFrame:
    """Thin a time-series DataFrame to at most `max_points` rows."""
    if df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _combine_date_time(
    df: pd.DataFrame,
    date_col: Optional[str],
    time_col: Optional[str],
    new_col: str,
) -> pd.DataFrame:
    """Create a datetime column from date + time (if available)."""
    df = df.copy()
    if date_col and date_col in df.columns and time_col and time_col in df.columns:
        df[new_col] = pd.to_datetime(
            df[date_col].astype(str) + " " + df[time_col].astype(str),
            errors="coerce",
        )
    elif date_col and date_col in df.columns:
        df[new_col] = pd.to_datetime(df[date_col], errors="coerce")
    elif time_col and time_col in df.columns:
        df[new_col] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df[new_col] = pd.NaT
    return df


def _safe_get_table(stay_data: Dict[str, Any], group: str, name: str) -> pd.DataFrame:
    """Helper to pull a DataFrame out of stay_data['icu'] or stay_data['hosp']."""
    group_dict = stay_data.get(group, {})
    if not isinstance(group_dict, dict):
        return pd.DataFrame()
    table = group_dict.get(name)
    if table is None:
        return pd.DataFrame()
    return table.copy()


# ---------------------------------------------------------------------
# TIME-SERIES PLOTS
# ---------------------------------------------------------------------

def render_medications_visuals(
    stay_data: Dict[str, Any], max_labels: int = 5, icu_intime: pd.Timestamp = None
) -> None:
    """
    For ICU medications: Plot Amount vs Hours since admission.
    """
    meds = _safe_get_table(stay_data, "icu", "medications")
    if meds.empty:
        st.info("No ICU medication records available.")
        return

    # [Existing logic to find columns...]
    label_col = "medications_label"
    amount_col = "medications_amount"
    
    # Combined date/time logic
    meds = _combine_date_time(
        meds,
        date_col="medications_start_date" if "medications_start_date" in meds.columns else None,
        time_col="medications_start_time" if "medications_start_time" in meds.columns else None,
        new_col="med_start_dt",
    )
    meds = meds.dropna(subset=["med_start_dt", amount_col])

    if meds.empty:
        return

    # Filter labels
    label_counts = meds[label_col].value_counts()
    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    selected_label = st.selectbox("Choose medication", all_labels, key="meds_select")

    df_label = meds[meds[label_col] == selected_label].copy()
    df_label = df_label.sort_values("med_start_dt")
    df_label = _downsample(df_label)

    # --- NEW X-AXIS LOGIC ---
    df_label["time"] = df_label["med_start_dt"]
    df_label["value"] = pd.to_numeric(df_label[amount_col], errors="coerce")
    
    x_axis_def = alt.X("time:T", title="Start Time")
    
    if icu_intime is not None:
        df_label["hours_since_admit"] = (df_label["time"] - icu_intime).dt.total_seconds() / 3600.0
        x_axis_def = alt.X("hours_since_admit:Q", title="Hours since ICU Admission")

    if alt is not None:
        chart = (
            alt.Chart(df_label)
            .mark_line(point=True)
            .encode(
                x=x_axis_def,
                y=alt.Y("value:Q", title="Amount"),
                tooltip=["time:T", "value:Q"]
            )
            .properties(title=f"Medication: {selected_label}", height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(df_label.set_index("time")["value"])


"""
def render_medications_visuals(stay_data: Dict[str, Any], max_labels: int = 5) -> None:
    
    For ICU medications:
    - Group by medications_label
    - Let user pick a label
    - Plot medications_amount over time (start datetime on x-axis)
    
    meds = _safe_get_table(stay_data, "icu", "medications")
    if meds.empty:
        st.info("No ICU medication records available for this stay.")
        return

    if "medications_label" not in meds.columns or "medications_amount" not in meds.columns:
        st.info("Medications table does not have label and amount columns.")
        return

    meds = _combine_date_time(
        meds,
        date_col="medications_start_date" if "medications_start_date" in meds.columns else None,
        time_col="medications_start_time" if "medications_start_time" in meds.columns else None,
        new_col="med_start_dt",
    )
    meds = meds.dropna(subset=["med_start_dt", "medications_amount"])
    if meds.empty:
        st.info("No medication records with valid start time and amount.")
        return

    # Choose label
    label_col = "medications_label"
    amount_col = "medications_amount"

    label_counts = meds[label_col].value_counts()
    if label_counts.empty:
        st.info("No medication labels to display.")
        return

    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    default_idx = 0
    if top_labels:
        default_label = top_labels[0]
        default_idx = all_labels.index(default_label)

    selected_label = st.selectbox(
        "Choose a medication to plot",
        all_labels,
        index=default_idx,
        key="meds_label_select",
    )

    df_label = meds[meds[label_col] == selected_label].copy()
    df_label = df_label.sort_values("med_start_dt")
    df_label = _downsample(df_label)

    if df_label.empty:
        st.info(f"No records found for medication '{selected_label}'.")
        return

    df_label["time"] = df_label["med_start_dt"]
    df_label["value"] = pd.to_numeric(df_label[amount_col], errors="coerce")
    df_label = df_label.dropna(subset=["value"])

    if df_label.empty:
        st.info(f"No numeric dose values to plot for '{selected_label}'.")
        return

    if alt is not None:
        chart = (
            alt.Chart(df_label)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", title="Start time"),
                y=alt.Y("value:Q", title="Amount"),
                tooltip=["time:T", "value:Q"],
            )
            .properties(title=f"Medication: {selected_label}", height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart_df = df_label.set_index("time")[["value"]]
        st.line_chart(chart_df)
"""


def render_measurements_visuals(
    stay_data: Dict[str, Any], max_labels: int = 5, icu_intime: pd.Timestamp = None
) -> None:
    """
    For ICU measurements: Plot value vs Hours since admission.
    """
    meas = _safe_get_table(stay_data, "icu", "measurements")
    if meas.empty:
        st.info("No ICU measurements available.")
        return

    # [Same label/column identification logic as before...]
    # ... (Keep existing logic to find label_col, val_col, time_col) ...
    # Assuming standard names for brevity in this snippet:
    label_col = "measurements_label"
    val_col = "measurements_valuenum"
    if "measurements_charttime" in meas.columns:
        meas["meas_time"] = pd.to_datetime(meas["measurements_charttime"], errors="coerce")
    else:
        return

    # Filter labels (Same logic as before)
    label_counts = meas[label_col].value_counts()
    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    selected_label = st.selectbox("Choose measurement", all_labels, key="meas_select")

    df_label = meas[meas[label_col] == selected_label].copy()
    df_label = df_label.sort_values("meas_time")
    df_label = _downsample(df_label)

    # --- NEW X-AXIS LOGIC ---
    df_label["time"] = df_label["meas_time"]
    df_label["value"] = pd.to_numeric(df_label[val_col], errors="coerce")
    
    x_axis_def = alt.X("time:T", title="Time")
    
    if icu_intime is not None:
        # Calculate hours since admission
        df_label["hours_since_admit"] = (df_label["time"] - icu_intime).dt.total_seconds() / 3600.0
        x_axis_def = alt.X("hours_since_admit:Q", title="Hours since ICU Admission")

    if alt is not None:
        chart = (
            alt.Chart(df_label)
            .mark_line(point=True)
            .encode(
                x=x_axis_def, # Use relative time if available
                y=alt.Y("value:Q", title="Value"),
                tooltip=["time:T", "value:Q", "hours_since_admit:Q"] if icu_intime else ["time:T", "value:Q"],
            )
            .properties(title=f"{selected_label}", height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(df_label.set_index("time")["value"])



"""
def render_measurements_visuals(
    stay_data: Dict[str, Any], max_labels: int = 5
) -> None:
    
    For ICU measurements:
    - Optionally filter to param_type == 'Numeric' if available
    - Group by measurements_label
    - Let user pick a label
    - Plot measurements_valuenum vs measurements_charttime
    
    meas = _safe_get_table(stay_data, "icu", "measurements")
    if meas.empty:
        st.info("No ICU measurements available for this stay.")
        return

    if "measurements_label" not in meas.columns or "measurements_valuenum" not in meas.columns:
        st.info("Measurements table does not have label and numeric value columns.")
        return

    # Optional filter to numeric param_type
    if "measurements_param_type" in meas.columns:
        meas = meas[meas["measurements_param_type"] == "Numeric"].copy()

    if "measurements_charttime" in meas.columns:
        meas["meas_time"] = pd.to_datetime(meas["measurements_charttime"], errors="coerce")
    else:
        st.info("Measurements table does not have a chart time column.")
        return

    meas = meas.dropna(subset=["meas_time", "measurements_valuenum"])
    if meas.empty:
        st.info("No numeric measurement values with valid timestamps.")
        return

    label_col = "measurements_label"
    value_col = "measurements_valuenum"

    label_counts = meas[label_col].value_counts()
    if label_counts.empty:
        st.info("No measurement labels to display.")
        return

    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    default_idx = 0
    if top_labels:
        default_label = top_labels[0]
        default_idx = all_labels.index(default_label)

    selected_label = st.selectbox(
        "Choose a measurement to plot",
        all_labels,
        index=default_idx,
        key="meas_label_select",
    )

    df_label = meas[meas[label_col] == selected_label].copy()
    df_label = df_label.sort_values("meas_time")
    df_label = _downsample(df_label)

    if df_label.empty:
        st.info(f"No records found for measurement '{selected_label}'.")
        return

    df_label["time"] = df_label["meas_time"]
    df_label["value"] = pd.to_numeric(df_label[value_col], errors="coerce")
    df_label = df_label.dropna(subset=["value"])

    if df_label.empty:
        st.info(f"No numeric values to plot for measurement '{selected_label}'.")
        return

    if alt is not None:
        chart = (
            alt.Chart(df_label)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("value:Q", title="Value"),
                tooltip=["time:T", "value:Q"],
            )
            .properties(title=f"Measurement: {selected_label}", height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart_df = df_label.set_index("time")[["value"]]
        st.line_chart(chart_df)
"""

def render_outputs_visuals(stay_data: Dict[str, Any], max_labels: int = 5) -> None:
    """
    For ICU outputevents:
    - Group by outputevents_label
    - Let user pick a label
    - Plot outputevents_value over outputevents_charttime
    """
    outs = _safe_get_table(stay_data, "icu", "outputevents")
    if outs.empty:
        st.info("No ICU output events available for this stay.")
        return

    if "outputevents_label" not in outs.columns or "outputevents_value" not in outs.columns:
        st.info("Outputevents table does not have label and value columns.")
        return

    # Prefer charttime; fall back to date+time if needed
    if "outputevents_charttime" in outs.columns:
        outs["out_time"] = pd.to_datetime(outs["outputevents_charttime"], errors="coerce")
    else:
        outs = _combine_date_time(
            outs,
            date_col="outputevents_date" if "outputevents_date" in outs.columns else None,
            time_col="outputevents_time" if "outputevents_time" in outs.columns else None,
            new_col="out_time",
        )

    outs = outs.dropna(subset=["out_time", "outputevents_value"])
    if outs.empty:
        st.info("No output values with valid timestamps.")
        return

    label_col = "outputevents_label"
    value_col = "outputevents_value"

    label_counts = outs[label_col].value_counts()
    if label_counts.empty:
        st.info("No output event labels to display.")
        return

    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    default_idx = 0
    if top_labels:
        default_label = top_labels[0]
        default_idx = all_labels.index(default_label)

    selected_label = st.selectbox(
        "Choose an output event to plot",
        all_labels,
        index=default_idx,
        key="outputs_label_select",
    )

    df_label = outs[outs[label_col] == selected_label].copy()
    df_label = df_label.sort_values("out_time")
    df_label = _downsample(df_label)

    if df_label.empty:
        st.info(f"No records found for output event '{selected_label}'.")
        return

    df_label["time"] = df_label["out_time"]
    df_label["value"] = pd.to_numeric(df_label[value_col], errors="coerce")
    df_label = df_label.dropna(subset=["value"])

    if df_label.empty:
        st.info(f"No numeric values to plot for output '{selected_label}'.")
        return

    if alt is not None:
        chart = (
            alt.Chart(df_label)
            .mark_line(point=True)
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("value:Q", title="Value"),
                tooltip=["time:T", "value:Q"],
            )
            .properties(title=f"Output event: {selected_label}", height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart_df = df_label.set_index("time")[["value"]]
        st.line_chart(chart_df)


def render_labs_visuals(
    stay_data: Dict[str, Any], max_labels: int = 5, icu_intime: pd.Timestamp = None
) -> None:
    labs = _safe_get_table(stay_data, "hosp", "labs")
    if labs.empty:
        st.info("No labs available.")
        return

    # [Keep existing column finding logic]
    if "lab_tests_charttime" in labs.columns:
        labs["lab_time"] = pd.to_datetime(labs["lab_tests_charttime"], errors="coerce")
    else:
        return

    # Filter labels logic...
    label_counts = labs["lab_tests_label"].value_counts()
    all_labels = list(label_counts.index)
    selected_label = st.selectbox("Choose lab test", all_labels, key="labs_select")

    df_label = labs[labs["lab_tests_label"] == selected_label].copy()
    df_label = df_label.sort_values("lab_time")
    df_label = _downsample(df_label)

    # --- NEW X-AXIS LOGIC ---
    df_label["time"] = df_label["lab_time"]
    df_label["value"] = pd.to_numeric(df_label["lab_tests_valuenum"], errors="coerce")
    
    x_axis_def = alt.X("time:T", title="Date/Time")
    
    if icu_intime is not None:
        df_label["hours_since_admit"] = (df_label["time"] - icu_intime).dt.total_seconds() / 3600.0
        x_axis_def = alt.X("hours_since_admit:Q", title="Hours since ICU Admission")

    if alt is not None:
        base = alt.Chart(df_label).encode(
            x=x_axis_def,
            y=alt.Y("value:Q", title="Value"),
             tooltip=["time:T", "value:Q"]
        )
        line = base.mark_line()
        points = base.mark_circle(size=60).encode(
            color=alt.condition(
                alt.datum.lab_tests_warning == 1, 
                alt.value("red"), 
                alt.value("steelblue")
            )
        )
        st.altair_chart((line + points).interactive(), use_container_width=True)
    else:
        st.line_chart(df_label.set_index("time")["value"])


"""
def render_labs_visuals(stay_data: Dict[str, Any], max_labels: int = 5) -> None:
    
    For HOSP lab_tests (ICU-window, cohort version):
    - Group by lab_tests_label
    - Let user pick a label
    - Plot lab_tests_valuenum over lab_tests_date (or charttime)
    - Highlight abnormal (warning == 1) points in red if Altair is available
    
    labs = _safe_get_table(stay_data, "hosp", "labs")
    if labs.empty:
        st.info("No lab tests available for this admission / ICU stay.")
        return

    needed_cols = {"lab_tests_label", "lab_tests_valuenum"}
    if not needed_cols.issubset(set(labs.columns)):
        st.info("Lab tests table does not have required label and numeric columns.")
        return

    # Time axis: use lab_tests_date if present, else charttime
    if "lab_tests_date" in labs.columns:
        labs["lab_time"] = pd.to_datetime(labs["lab_tests_date"], errors="coerce")
    elif "lab_tests_charttime" in labs.columns:
        labs["lab_time"] = pd.to_datetime(labs["lab_tests_charttime"], errors="coerce")
    else:
        st.info("Lab tests table does not have a usable time column.")
        return

    labs["lab_tests_valuenum"] = pd.to_numeric(labs["lab_tests_valuenum"], errors="coerce")
    labs = labs.dropna(subset=["lab_time", "lab_tests_valuenum"])
    if labs.empty:
        st.info("No numeric lab values with valid timestamps.")
        return

    label_col = "lab_tests_label"
    value_col = "lab_tests_valuenum"

    label_counts = labs[label_col].value_counts()
    if label_counts.empty:
        st.info("No lab test labels to display.")
        return

    top_labels = label_counts.head(max_labels).index.tolist()
    all_labels = list(label_counts.index)
    default_idx = 0
    if top_labels:
        default_label = top_labels[0]
        default_idx = all_labels.index(default_label)

    selected_label = st.selectbox(
        "Choose a lab test to plot",
        all_labels,
        index=default_idx,
        key="labs_label_select",
    )

    df_label = labs[labs[label_col] == selected_label].copy()
    df_label = df_label.sort_values("lab_time")
    df_label = _downsample(df_label)

    if df_label.empty:
        st.info(f"No records found for lab test '{selected_label}'.")
        return

    df_label["time"] = df_label["lab_time"]
    df_label["value"] = df_label[value_col]
    df_label["is_abnormal"] = (
        df_label.get("lab_tests_warning", 0).fillna(0).astype(int)
    )

    if alt is not None:
        base = alt.Chart(df_label).encode(
            x=alt.X("time:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
        )
        line = base.mark_line()
        abnormal_points = base.transform_filter(
            alt.datum.is_abnormal == 1
        ).mark_circle(size=60, color="red")
        chart = (line + abnormal_points).properties(
            title=f"Lab test: {selected_label}", height=300
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart_df = df_label.set_index("time")[["value"]]
        st.line_chart(chart_df)
"""

# ---------------------------------------------------------------------
# STRUCTURED TEXT TABLES
# ---------------------------------------------------------------------


def render_admission_table(stay_data: Dict[str, Any]) -> None:
    """Structured view of patients_admissions_clean_icu_250 row."""
    hosp = stay_data.get("hosp", {})
    patadm = hosp.get("patients_admissions")
    if patadm is None or patadm.empty:
        st.info("No admission row available for this stay.")
        return

    row = patadm.iloc[0]
    fields = [
        "gender",
        "anchor_age",
        "admission_type",
        "admission_location",
        "admittime",
        "dischtime",
        "discharge_location",
        "deathtime",
        "deathdate",
        "hospital_expire_flag",
    ]

    display_rows: List[Dict[str, Any]] = []
    for f in fields:
        if f in row.index:
            val = row[f]
            if pd.notna(val):
                display_rows.append({"Field": f, "Value": val})

    if not display_rows:
        st.info("No non-null admission fields to display.")
        return

    df_disp = pd.DataFrame(display_rows)
    st.table(df_disp)


def render_diagnoses_table(stay_data: Dict[str, Any]) -> None:
    """Ordered diagnoses for this hadm_id from diagnoses_clean_icu_250."""
    hosp = stay_data.get("hosp", {})
    dx = hosp.get("diagnoses")
    if dx is None or dx.empty:
        st.info("No diagnoses found for this admission.")
        return

    df_dx = dx.copy()
    if "dx_seq_num" in df_dx.columns:
        df_dx = df_dx.sort_values("dx_seq_num")

    cols = []
    if "dx_seq_num" in df_dx.columns:
        cols.append("dx_seq_num")
    if "dx_long_title" in df_dx.columns:
        cols.append("dx_long_title")

    if not cols:
        st.info("Diagnoses table does not have expected columns.")
        return

    df_disp = df_dx[cols].rename(
        columns={
            "dx_seq_num": "Sequence",
            "dx_long_title": "Diagnosis",
        }
    )
    st.table(df_disp)


def render_hosp_procedures_table(stay_data: Dict[str, Any]) -> None:
    """Ordered HOSP procedures (ICU-window filtered) for this stay/hadm."""
    hosp = stay_data.get("hosp", {})
    procs = hosp.get("procedures")
    if procs is None or procs.empty:
        st.info("No HOSP procedures for this ICU stay.")
        return

    df_p = procs.copy()
    if "proc_seq_num" in df_p.columns:
        df_p = df_p.sort_values("proc_seq_num")

    cols = []
    if "proc_seq_num" in df_p.columns:
        cols.append("proc_seq_num")
    if "proc_long_title" in df_p.columns:
        cols.append("proc_long_title")
    time_col = None
    if "procedure_chartdatetime" in df_p.columns:
        cols.append("procedure_chartdatetime")
        time_col = "procedure_chartdatetime"
    elif "procedure_date" in df_p.columns:
        cols.append("procedure_date")
        time_col = "procedure_date"

    df_disp = df_p[cols].rename(
        columns={
            "proc_seq_num": "Sequence",
            "proc_long_title": "Procedure",
            "procedure_chartdatetime": "Time",
            "procedure_date": "Date",
        }
    )
    st.table(df_disp)


def render_icu_procedureevents_table(stay_data: Dict[str, Any]) -> None:
    """ICU procedureevents (bedside procedures) for this stay."""
    icu = stay_data.get("icu", {})
    pe = icu.get("procedureevents")
    if pe is None or pe.empty:
        st.info("No ICU procedureevents found for this stay.")
        return

    df = pe.copy()
    # Build a simple start datetime from start_date if available
    df = _combine_date_time(
        df,
        date_col="procedureevents_start_date"
        if "procedureevents_start_date" in df.columns
        else None,
        time_col=None,  # cohort summary only shows start_date / end_date
        new_col="proc_start_dt",
    )

    cols = []
    if "procedureevents_category" in df.columns:
        cols.append("procedureevents_category")
    if "procedureevents_label" in df.columns:
        cols.append("procedureevents_label")
    if "procedureevents_location" in df.columns:
        cols.append("procedureevents_location")
    if "procedureevents_value" in df.columns:
        cols.append("procedureevents_value")
    if "procedureevents_valueuom" in df.columns:
        cols.append("procedureevents_valueuom")
    if "proc_start_dt" in df.columns:
        cols.append("proc_start_dt")

    df_disp = df[cols].rename(
        columns={
            "procedureevents_category": "Category",
            "procedureevents_label": "Procedure",
            "procedureevents_location": "Location",
            "procedureevents_value": "Value",
            "procedureevents_valueuom": "Unit",
            "proc_start_dt": "Start date",
        }
    ).sort_values("Start date")

    st.table(df_disp)