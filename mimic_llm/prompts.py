from typing import Any, Dict, List


# --------------------------------------------------------------------
# Helper formatters: turn structured dicts into readable bullet blocks
# --------------------------------------------------------------------

def _format_demographics(demo: Dict[str, Any]) -> str:
    """Format demographics + admission context into a short text block."""
    if not demo:
        return (
            "Patient demographics and admission context:\n"
            "- (No demographic or admission information available.)\n\n"
        )

    age = demo.get("age") or demo.get("anchor_age") or demo.get("admission_age")
    try:
        age_int = int(age) if age is not None else None
    except Exception:
        age_int = None

    gender = demo.get("gender") or "Unknown"
    admission_type = demo.get("admission_type") or "Unknown"
    admission_location = demo.get("admission_location") or "Unknown"
    discharge_location = demo.get("discharge_location") or "Unknown"

    admit_time = (
        demo.get("admittime")
        or demo.get("admit_datetime")
        or demo.get("admitdatetime")
        or demo.get("admission_time")
    )
    discharge_time = (
        demo.get("dischtime")
        or demo.get("discharge_datetime")
        or demo.get("dischdatetime")
    )
    death_time = (
        demo.get("deathtime")
        or demo.get("death_datetime")
        or demo.get("deathdate")
    )
    hosp_expire_flag = demo.get("hospital_expire_flag")

    if age_int is not None and age_int >= 0:
        age_str = f"{age_int}"
    else:
        age_str = "Unknown"

    lines = ["Patient demographics and admission context:"]
    lines.append(f"- Age: {age_str}")
    lines.append(f"- Gender: {gender}")
    lines.append(f"- Admission type: {admission_type}")
    lines.append(f"- Admitted from: {admission_location}")
    if admit_time:
        lines.append(f"- Admission time: {admit_time}")
    lines.append(f"- Discharged to: {discharge_location}")
    if discharge_time:
        lines.append(f"- Discharge time: {discharge_time}")

    # Outcome line: discharged vs died vs unknown
    if (hosp_expire_flag == 1) or death_time:
        if death_time:
            lines.append(f"- Outcome: patient died during this admission (time: {death_time})")
        else:
            lines.append("- Outcome: patient died during this admission (death time not recorded).")
    else:
        # Only say "discharged" if we have some evidence of discharge info
        if discharge_time or discharge_location != "Unknown":
            lines.append("- Outcome: patient was discharged from this admission.")
        else:
            lines.append("- Outcome: admission outcome not recorded in this data.")

    lines.append("")
    return "\n".join(lines)


def _format_diagnoses(dx_list: List[Dict[str, Any]], max_n: int = 10) -> str:
    """Format diagnoses list into an ordered bullet block."""
    if not dx_list:
        return "Diagnoses during this hospital admission:\n- (No diagnoses recorded.)\n\n"

    # Try to sort by explicit sequence if present
    def _seq(row: Dict[str, Any]) -> Any:
        return (
            row.get("dx_seq_num")
            or row.get("sequence")
            or row.get("seq_num")
            or 0
        )

    sorted_dx = sorted(dx_list, key=_seq)
    sorted_dx = sorted_dx[:max_n]

    lines = ["Diagnoses during this hospital admission (ordered):"]
    for i, dx in enumerate(sorted_dx, start=1):
        title = (
            dx.get("dx_long_title")
            or dx.get("long_title")
            or dx.get("title")
            or dx.get("icd_code")
            or "Unknown diagnosis"
        )
        lines.append(f"{i}. {title}")
    lines.append("")
    return "\n".join(lines)


def _format_procedures(proc_list: List[Dict[str, Any]], max_n: int = 10) -> str:
    """Format procedures (usually HOSP procedures filtered to ICU window)."""
    if not proc_list:
        return (
            "Procedures performed during this admission (ICU-relevant window):\n"
            "- (No procedures recorded in the data for this window.)\n\n"
        )

    def _time_str(row: Dict[str, Any]) -> str:
        return (
            str(row.get("procedure_chartdatetime"))
            or str(row.get("charttime_str") or row.get("charttime") or "")
        )

    lines = ["Procedures performed during this admission (ICU-relevant window):"]
    for i, proc in enumerate(proc_list[:max_n], start=1):
        name = (
            proc.get("proc_long_title")
            or proc.get("procedure_name")
            or proc.get("label")
            or "Unknown procedure"
        )
        when = _time_str(proc)
        if when and when != "None":
            lines.append(f"{i}. {name} (around {when})")
        else:
            lines.append(f"{i}. {name}")
    lines.append("")
    return "\n".join(lines)


def _format_labs(lab_rows: List[Dict[str, Any]]) -> str:
    """Format aggregated lab summary rows."""
    if not lab_rows:
        return (
            "Key laboratory results and trends during the ICU stay:\n"
            "- (No ICU lab results available in the data.)\n\n"
        )

    lines = [
        "Key laboratory results and trends during the ICU stay "
        "(each bullet summarises one lab test):"
    ]
    for row in lab_rows:
        name = (
            row.get("lab_name")
            or row.get("lab_tests_label")
            or row.get("label")
            or row.get("itemid")
            or "Unknown lab test"
        )
        
        # Attempt to get unit from the new field first, fallback to old keys
        unit = (
            row.get("unit")
            or row.get("valueuom")
            or row.get("lab_tests_valueuom")
            or row.get("unitname")
            or ""
        )
        
        low = row.get("min")
        med = row.get("median")
        high = row.get("max")
        count = row.get("count")
        abn = row.get("abnormal_count")
        trend = row.get("trend")

        parts = [f"- {name}"]
        # We append unit to the numbers now, rather than the name
        
        stats_bits = []
        if med is not None:
            val_str = f"{med:.3g}"
            if unit:
                val_str += f" {unit}"
            stats_bits.append(f"median {val_str}")
            
        if low is not None and high is not None:
            range_str = f"{low:.3g}–{high:.3g}"
            if unit:
                range_str += f" {unit}"
            stats_bits.append(f"range {range_str}")
            
        # Add Trend
        if trend and trend not in ["Unknown", "Insufficient data", "Stable"]:
            stats_bits.append(f"trend: {trend.lower()}")
        elif trend == "Stable":
            stats_bits.append("trend: stable")

        if stats_bits:
            parts.append(" with " + ", ".join(stats_bits))
            
        if count is not None:
            parts.append(f"; n={int(count)}")
        if abn is not None and abn > 0:
            parts.append(f", abnormal results: {int(abn)}")
            
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)


def _format_meds(meds_rows: List[Dict[str, Any]]) -> str:
    """Format aggregated ICU medications summary rows."""
    if not meds_rows:
        return (
            "ICU medications summary (grouped by label/category):\n"
            "- (No ICU medications recorded in the data.)\n\n"
        )

    lines = ["ICU medications summary (each bullet summarises one medication label):"]
    for row in meds_rows:
        name = (
            row.get("med_name")
            or row.get("medications_label")
            or row.get("drug_name")
            or row.get("label")
            or "Unknown medication"
        )
        category = row.get("category")
        n_orders = row.get("num_orders")
        total_amount = row.get("total_amount")
        unit = row.get("unit") or ""  # <--- Get Unit
        
        start = row.get("first_start") or row.get("start_time") or row.get("start")
        end = row.get("last_end") or row.get("end_time") or row.get("end")

        parts = [f"- {name}"]
        if category:
            parts[-1] += f" [{category}]"
        if n_orders is not None:
            parts.append(f", number of orders: {int(n_orders)}")
        
        if total_amount is not None:
            amt_str = f"{float(total_amount):.3g}"
            if unit:
                amt_str += f" {unit}"  # <--- Append Unit
            parts.append(f", approximate total amount: {amt_str}")
            
        if start or end:
            parts.append(" (given")
            if start:
                parts.append(f" from {start}")
            if end:
                parts.append(f" to {end}")
            parts.append(")")
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)

"""
def _format_meds(meds_rows: List[Dict[str, Any]]) -> str:
    Format aggregated ICU medications summary rows.
    if not meds_rows:
        return (
            "ICU medications summary (grouped by label/category):\n"
            "- (No ICU medications recorded in the data.)\n\n"
        )

    lines = ["ICU medications summary (each bullet summarises one medication label):"]
    for row in meds_rows:
        name = (
            row.get("med_name")
            or row.get("medications_label")
            or row.get("drug_name")
            or row.get("label")
            or "Unknown medication"
        )
        category = row.get("category")
        n_orders = row.get("num_orders")
        total_amount = row.get("total_amount")
        start = row.get("first_start") or row.get("start_time") or row.get("start")
        end = row.get("last_end") or row.get("end_time") or row.get("end")

        parts = [f"- {name}"]
        if category:
            parts[-1] += f" [{category}]"
        if n_orders is not None:
            parts.append(f", number of orders: {int(n_orders)}")
        if total_amount is not None:
            parts.append(f", approximate total amount: {float(total_amount):.3g}")
        if start or end:
            parts.append(" (given")
            if start:
                parts.append(f" from {start}")
            if end:
                parts.append(f" to {end}")
            parts.append(")")
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)
"""

def _format_measurements(meas_rows: List[Dict[str, Any]]) -> str:
    """Format aggregated ICU measurements summary rows."""
    if not meas_rows:
        return (
            "Summarised bedside measurements and vital-sign trends during the ICU stay:\n"
            "- (No ICU measurements available in the data.)\n\n"
        )

    lines = [
        "Summarised bedside measurements and vital-sign trends during the ICU stay "
        "(each bullet summarises one measurement label):"
    ]
    for row in meas_rows:
        name = (
            row.get("measure_name")
            or row.get("measurements_label")
            or row.get("label")
            or "Unknown measurement"
        )
        low = row.get("min")
        med = row.get("median")
        high = row.get("max")
        count = row.get("count")
        
        # New fields
        unit = row.get("unit") or ""
        trend = row.get("trend")

        parts = [f"- {name}"]
        stats_bits = []
        
        # Format Median with Unit
        if med is not None:
            val_str = f"{med:.3g}"
            if unit:
                val_str += f" {unit}"
            stats_bits.append(f"median {val_str}")
        
        # Format Range with Unit
        if low is not None and high is not None:
            range_str = f"{low:.3g}–{high:.3g}"
            if unit:
                range_str += f" {unit}"
            stats_bits.append(f"range {range_str}")
            
        # Format Trend
        if trend and trend not in ["Unknown", "Insufficient data", "Stable"]:
            # Only mention trend if it's Rising or Falling to save tokens, 
            # or you can include "stable" if preferred.
            stats_bits.append(f"trend: {trend.lower()}")
        elif trend == "Stable":
            stats_bits.append("trend: stable")

        if stats_bits:
            parts.append(" with " + ", ".join(stats_bits))
        if count is not None:
            parts.append(f"; n={int(count)}")
            
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)

"""
def _format_measurements(meas_rows: List[Dict[str, Any]]) -> str:
    Format aggregated ICU measurements summary rows.
    if not meas_rows:
        return (
            "Summarised bedside measurements and vital-sign trends during the ICU stay:\n"
            "- (No ICU measurements available in the data.)\n\n"
        )

    lines = [
        "Summarised bedside measurements and vital-sign trends during the ICU stay "
        "(each bullet summarises one measurement label):"
    ]
    for row in meas_rows:
        name = (
            row.get("measure_name")
            or row.get("measurements_label")
            or row.get("label")
            or "Unknown measurement"
        )
        low = row.get("min")
        med = row.get("median")
        high = row.get("max")
        count = row.get("count")

        parts = [f"- {name}"]
        stats_bits = []
        if med is not None:
            stats_bits.append(f"median {med:.3g}")
        if low is not None and high is not None:
            stats_bits.append(f"range {low:.3g}–{high:.3g}")
        if stats_bits:
            parts.append(" with " + ", ".join(stats_bits))
        if count is not None:
            parts.append(f"; n={int(count)}")
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)
"""

def _format_outputs(outputs_rows: List[Dict[str, Any]]) -> str:
    """Format aggregated ICU output events summary rows."""
    if not outputs_rows:
        return (
            "ICU output events (urine, drains, etc.):\n"
            "- (No ICU output events recorded in the data.)\n\n"
        )

    lines = [
        "ICU output events (each bullet summarises one output label over the ICU stay):"
    ]
    for row in outputs_rows:
        name = (
            row.get("output_label")
            or row.get("outputevents_label")
            or row.get("label")
            or "Unknown output"
        )
        unit = (
            row.get("unit")
            or row.get("outputevents_valueuom")
            or row.get("valueuom")
            or ""
        )
        total = row.get("total_volume") or row.get("sum")
        low = row.get("min")
        med = row.get("median")
        high = row.get("max")
        count = row.get("count")

        parts = [f"- {name}"]
        if unit:
            parts[-1] += f" ({unit})"
        stats_bits = []
        if total is not None:
            stats_bits.append(f"total ~{float(total):.3g}")
        if med is not None:
            stats_bits.append(f"median {med:.3g}")
        if low is not None and high is not None:
            stats_bits.append(f"range {low:.3g}–{high:.3g}")
        if stats_bits:
            parts.append(" with " + ", ".join(stats_bits))
        if count is not None:
            parts.append(f"; n={int(count)}")
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)


def _format_procedureevents(proc_ev_rows: List[Dict[str, Any]]) -> str:
    """Format aggregated ICU procedureevents summary rows."""
    if not proc_ev_rows:
        return (
            "ICU bedside procedures and interventions:\n"
            "- (No ICU bedside procedures recorded in the data.)\n\n"
        )

    lines = [
        "ICU bedside procedures and interventions "
        "(each bullet summarises one procedureevents label):"
    ]
    for row in proc_ev_rows:
        label = (
            row.get("procedureevents_label")
            or row.get("label")
            or "Unknown procedure"
        )
        category = row.get("category") or row.get("procedureevents_category")
        location = row.get("location") or row.get("procedureevents_location")
        start = row.get("start") or row.get("procedureevents_startdatetime")
        end = row.get("end") or row.get("procedureevents_enddatetime")

        parts = [f"- {label}"]
        if category:
            parts[-1] += f" [{category}]"
        if location:
            parts.append(f" at {location}")
        if start or end:
            parts.append(" (performed")
            if start:
                parts.append(f" from {start}")
            if end:
                parts.append(f" to {end}")
            parts.append(")")
        lines.append("".join(parts))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------
# FLAN-style prompts (instruction-heavy)
# --------------------------------------------------------------------

def _make_flan_prompt(view_type: str, features: Dict[str, Any]) -> str:
    """Build an instruction-style prompt for FLAN-T5."""
    vt = (view_type or "").lower()

    demo_block = _format_demographics(features.get("demographics", {}))

    # 1) Admission & demographics view
    if vt == "admission":
        header = (
            "You are an ICU clinician writing a brief, factual admission note for another doctor.\n\n"
            "Task:\n"
            "- Using only the structured information below, write 2–3 sentences that describe:\n"
            "  * the patient's age and gender,\n"
            "  * how and when the patient was admitted,\n"
            "  * and whether the patient was discharged or died during this admission.\n\n"
            "Requirements:\n"
            "- Use only the information that appears in the structured data.\n"
            "- Do not guess or add clinical interpretation.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the numbers, only the structure):\n"
            "The patient is a 67-year-old woman admitted as an emergency from the emergency room on 2115-09-12. "
            "She was discharged home on 2115-09-20.\n\n"
        )
        body = "Structured data:\n" + demo_block + "\nNow write the summary:\n"
        return header + example + body

    # 2) Diagnoses + procedures view
    if vt == "dx_proc":
        dx_block = _format_diagnoses(features.get("diagnoses", []), max_n=5)
        proc_block = _format_procedures(
            features.get("icu_procedures", []) or features.get("procedures", []),
            max_n=5,
        )
        header = (
            "You are an ICU clinician writing a concise, factual summary of diagnoses and procedures "
            "for this hospital admission.\n\n"
            "Task:\n"
            "- Using only the information below, write 3–5 sentences that:\n"
            "  * first list up to 5 main diagnoses in order of importance,\n"
            "  * then list up to 5 key procedures performed during the admission.\n\n"
            "Requirements:\n"
            "- Use only the diagnosis and procedure names shown in the structured data.\n"
            "- Do not invent new diagnoses, procedures, or explanations.\n"
            "- Do not add causal statements or clinical interpretation beyond what is explicit.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the content, only the structure):\n"
            "Primary diagnoses included congestive heart failure and acute myocardial infarction of the anterolateral wall. "
            "Additional diagnoses included essential hypertension. "
            "Key procedures during this admission included coronary angiography and insertion of a pulmonary artery catheter early in the ICU stay. "
            "A tracheostomy was performed later during the admission.\n\n"
        )
        body = "Structured data:\n" + demo_block + dx_block + proc_block + "Now write the summary:\n"
        return header + example + body

    # 3) Lab events
    if vt == "labs":
        labs_block = _format_labs(features.get("labs_summary", []))
        header = (
            "You are an ICU clinician summarising the key laboratory results and trends for this ICU stay.\n\n"
            "Task:\n"
            "- Using only the laboratory information below, write 3–5 sentences that describe:\n"
            "  * which lab tests are most important,\n"
            "  * their typical values (medians and ranges),\n"
            "  * and specifically mention the 'trend' (Rising, Falling, or Stable) provided in the data for each test.\n\n"
            "Requirements:\n"
            "- Use only the tests, values, and trends shown in the structured data.\n"
            "- Do not invent new lab tests or values.\n"
            "- Do not provide detailed pathophysiological explanations; stay factual.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the numbers, only the structure):\n"
            "Key laboratory tests included creatinine with a median of 1.4 mg/dL (range 0.9–2.3) and a rising trend. "
            "Hemoglobin was repeatedly low with a median of 9.2 g/dL (range 8.5–10.0) and remained stable. "
            "Sodium levels were relatively stable around a median of 138 mmol/L.\n\n"
        )
        body = "Structured data:\n" + labs_block + "Now write the lab summary:\n"
        return header + example + body

    # 4) Medications
    if vt == "meds":
        meds_block = _format_meds(features.get("meds_summary", []))
        header = (
            "You are an ICU clinician summarising the medication course for this ICU stay.\n\n"
            "Task:\n"
            "- Using only the medication information below, write 3–4 sentences that:\n"
            "  * highlight the most important medications in each category,\n"
            "  * mention total amounts (with units) and time periods,\n"
            "  * and describe the overall therapeutic strategy (for example, antibiotics for infection or vasopressors for shock) "
            "without inventing new drugs.\n\n"
            "Requirements:\n"
            "- Use only the medication names, categories, and dates shown in the structured data.\n"
            "- Do not invent drug names, doses, or durations.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the drug names, only the structure):\n"
            "Key medications included norepinephrine given repeatedly from 2115-09-12 to 2115-09-14, "
            "and piperacillin–tazobactam as an antibiotic from 2115-09-12 to 2115-09-18.\n\n"
        )
        body = "Structured data:\n" + meds_block + "Now write the medication summary:\n"
        return header + example + body

    # 5) Measurements / vitals
    if vt == "measurements":
        meas_block = _format_measurements(features.get("measurements_summary", []))
        header = (
            "You are an ICU clinician summarising vital signs and other bedside measurements for this ICU stay.\n\n"
            "Task:\n"
            "- Using only the measurement information below, write 3–5 sentences that describe:\n"
            "  * the typical values (medians and ranges) for key measurements,\n"
            "  * and explicitly mention the 'trend' (Rising, Falling, or Stable) for each vital sign.\n\n"
            "Requirements:\n"
            "- Use only the measurement labels, values, and trends shown in the structured data.\n"
            "- Do not label values as 'normal' or 'abnormal' unless this is explicitly encoded; just describe the numbers.\n"
            "- Do not invent additional measurements or time periods.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the numbers, only the structure):\n"
            "During the ICU stay, oxygen saturation was generally well maintained with a median of 96% (range 90–99%). "
            "Heart rate showed a rising trend with a median of 92 beats per minute (range 70–130). "
            "Systolic blood pressure was stable with a median of 110 mmHg. "
            "Respiratory rate remained relatively stable around a median of 18 breaths per minute.\n\n"
        )
        body = "Structured data:\n" + meas_block + "Now write the measurements summary:\n"
        return header + example + body

    # 6) Output events
    if vt == "outputs":
        outputs_block = _format_outputs(features.get("outputs_summary", []))
        header = (
            "You are an ICU clinician summarising fluid outputs for this ICU stay.\n\n"
            "Task:\n"
            "- Using only the output information below, write 3–5 sentences that describe:\n"
            "  * the main types of outputs (for example, urine via Foley catheter, drain output),\n"
            "  * approximate total volumes and time windows when available,\n"
            "  * and simple trends such as stable, increasing, or decreasing outputs.\n\n"
            "Requirements:\n"
            "- Use only the output labels, units, and values shown in the structured data.\n"
            "- Do not invent additional fluids, volumes, or trends.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the numbers, only the structure):\n"
            "Urine output via Foley catheter totalled about 1800 mL between 2115-09-12 and 2115-09-13 and remained relatively stable. "
            "Chest drain output was around 600 mL from 2115-09-12 to 2115-09-14 and tended to decrease over time. "
            "Nasogastric output was modest with lower volumes and no clear trend. "
            "Overall, fluid outputs were monitored closely with stable urinary output and gradually declining drain output.\n\n"
        )
        body = "Structured data:\n" + outputs_block + "Now write the output events summary:\n"
        return header + example + body

    # 7) ICU procedureevents
    if vt == "procedureevents":
        proc_ev_block = _format_procedureevents(features.get("procedureevents_summary", []))
        header = (
            "You are an ICU clinician summarising bedside procedures and interventions during this ICU stay.\n\n"
            "Task:\n"
            "- Using only the procedure information below, write 3–5 sentences that describe:\n"
            "  * the types of procedures performed,\n"
            "  * where they were performed (location),\n"
            "  * and the approximate timing of these procedures.\n\n"
            "Requirements:\n"
            "- Use only the procedure labels, categories, locations, and times shown in the structured data.\n"
            "- Do not invent new procedures, indications, or complications.\n"
            "- Do not mention 'tables' or 'structured data'.\n\n"
        )
        example = (
            "Example output style (do NOT copy the content, only the structure):\n"
            "ICU bedside procedures included placement of a 20-gauge peripheral line in the left forearm on 2115-09-12. "
            "Chest X-rays were obtained in the ICU on 2115-09-12. "
            "A paracentesis was performed later during the ICU stay. "
            "These procedures were performed at the bedside in the ICU.\n\n"
        )
        body = "Structured data:\n" + proc_ev_block + "Now write the procedureevents summary:\n"
        return header + example + body

# --------------------------------------------------------------------
# Meditron-style prompts (shorter, completion-oriented)
# --------------------------------------------------------------------

def _make_meditron_prompt(view_type: str, features: Dict[str, Any]) -> str:
    """Build a Llama-2 style instruction prompt for Meditron."""
    vt = (view_type or "").lower()
    demo_block = _format_demographics(features.get("demographics", {}))

    # Base instruction wrapper
    def wrap_inst(instruction, data_content):
        return (
            "[INST] You are a helpful clinical assistant. "
            f"{instruction}\n\n"
            "Structured Data:\n"
            f"{data_content}\n"
            "[/INST]\n"
            "Summary:"
        )

    if vt == "admission":
        return wrap_inst(
            instruction=(
                "Using the structured admission data below, write 2–3 sentences describing "
                "the patient's age, gender, admission context, and whether they were discharged or died. "
                "Do not invent clinical details."
            ),
            data_content=f"{demo_block}"
        )

    if vt == "dx_proc":
        dx_block = _format_diagnoses(features.get("diagnoses", []), max_n=5)
        proc_block = _format_procedures(
            features.get("icu_procedures", []) or features.get("procedures", []),
            max_n=5,
        )
        return wrap_inst(
            instruction=(
                "Using the structured data below, write 3–5 sentences describing the main diagnoses "
                "and key procedures in the order given. Do not add extra interpretation."
            ),
            data_content=f"{demo_block}{dx_block}{proc_block}"
        )

    if vt == "labs":
        labs_block = _format_labs(features.get("labs_summary", []))
        return wrap_inst(
            instruction=(
                "Using the lab tests below, write 3–5 sentences describing key tests, "
                "their median values/ranges, and the calculated trend (Rising/Falling/Stable)."
            ),
            data_content=f"{labs_block}"
        )

    if vt == "meds":
        meds_block = _format_meds(features.get("meds_summary", []))
        return wrap_inst(
            instruction=(
                "Using the medication list below, write 3–4 sentences highlighting the most important "
                "medications in each category and their approximate time periods."
            ),
            data_content=f"{meds_block}"
        )

    if vt == "measurements":
        meas_block = _format_measurements(features.get("measurements_summary", []))
        return wrap_inst(
            instruction=(
                "Using the measurements below, write 3–5 sentences describing the main vitals, "
                "their median values/ranges, and their trend (Rising/Falling/Stable). "
                "Do not use labels like 'normal' unless explicitly shown."
            ),
            data_content=f"{meas_block}"
        )

    if vt == "outputs":
        outputs_block = _format_outputs(features.get("outputs_summary", []))
        return wrap_inst(
            instruction=(
                "Using the output events below (urine, drains, etc.), write 3–5 sentences describing "
                "the main output types, total volumes, and time windows."
            ),
            data_content=f"{outputs_block}"
        )

    if vt == "procedureevents":
        proc_ev_block = _format_procedureevents(features.get("procedureevents_summary", []))
        return wrap_inst(
            instruction=(
                "Using the ICU bedside procedures below, write 3–5 sentences describing the procedures "
                "by category, mentioning locations and approximate dates."
            ),
            data_content=f"{proc_ev_block}"
        )

    # Default fallback to dx_proc style if view_type is unknown
    return _make_meditron_prompt("dx_proc", features)

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def make_prompt(view_type: str, features: Dict[str, Any], model_name: str = "") -> str:
    """
    Build a text prompt for a given view and model.

    model_name:
      - "flan" or containing "flan"/"t5"  -> FLAN-style instruction prompt
      - "meditron"                        -> Meditron clinical-note style prompt
    """
    name = (model_name or "").lower()
    if "meditron" in name:
        return _make_meditron_prompt(view_type, features)
    if "flan" in name or "t5" in name:
        return _make_flan_prompt(view_type, features)
    # Default to FLAN style if unknown
    return _make_flan_prompt(view_type, features)