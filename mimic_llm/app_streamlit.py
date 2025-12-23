import os
import sys
import json
import pandas as pd
import streamlit as st

# Wire project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import COHORT_META_DIR
from features import load_all_tables_for_stay
from eval import compare_summaries  # Import for live metric calculation
from visuals import (
    render_medications_visuals,
    render_measurements_visuals,
    render_outputs_visuals,
    render_labs_visuals,
    render_admission_table,
    render_diagnoses_table,
    render_hosp_procedures_table,
    render_icu_procedureevents_table,
)

# Load the precomputed JSON
PRECOMPUTED_PATH = os.path.join(PROJECT_ROOT, "precomputed_cohort_summaries.json")

# Define available view options
VIEW_LABELS = {
    "admission": "Admission & Demographics",
    "dx_proc": "Diagnoses & Procedures",
    "labs": "Lab Events",
    "meds": "Medications",
    "measurements": "Vitals / Measurements",
    "outputs": "Outputs (Fluids)",
    "procedureevents": "ICU Procedures",
    "final": "Final Discharge Summary"
}
VIEW_KEYS = {v: k for k, v in VIEW_LABELS.items()}

@st.cache_data
def load_precomputed_data():
    if not os.path.exists(PRECOMPUTED_PATH):
        return {}
    with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def main():
    st.set_page_config(layout="wide") # Crucial for split screen
    st.title("ICU Discharge Summary Assistant")

    # 1. Load Data
    precomputed = load_precomputed_data()
    
    if not precomputed:
        st.error(f"Precomputed summaries file not found at: {PRECOMPUTED_PATH}. Please run scripts/precompute_summaries.py first.")
        return

    # Dropdown for Stay ID
    available_stays = sorted([int(k) for k in precomputed.keys()])
    
    # Sidebar
    with st.sidebar:
        st.header("Patient Selection")
        selected_stay_id = st.selectbox(
            "Select ICU Stay ID", 
            available_stays,
            index=0
        )
        
        st.markdown("---")
        st.header("View Selection")
        # View selector determines what shows on the RIGHT side
        view_label = st.selectbox(
            "Select Clinical View",
            list(VIEW_LABELS.values()),
            index=3 # Default to meds or measurements
        )
        current_view_slug = VIEW_KEYS[view_label]

        st.markdown("---")
        st.header("Display Options")
        show_ground_truth = st.checkbox("Show Actual Discharge Note", value=True)

    # Get data for selected stay
    stay_record = precomputed.get(str(selected_stay_id))
    
    # We still need to load the raw tables for Visuals (from disk)
    # This is fast enough to do on-demand (Parquet read)
    try:
        raw_stay_data = load_all_tables_for_stay(selected_stay_id)
        icu_intime = None
        if not raw_stay_data["icu"]["icustays"].empty:
             icu_intime = pd.to_datetime(raw_stay_data["icu"]["icustays"].iloc[0]["intime"])
    except Exception as e:
        st.error(f"Could not load raw data tables: {e}")
        return

    # -----------------------------------------------------------
    # LAYOUT: 2 Columns
    # Left: Global Context (Full Summary + Ground Truth)
    # Right: Specific Context (View Summary + Visuals)
    # -----------------------------------------------------------
    
    left_col, right_col = st.columns([1, 1], gap="large")

    # === LEFT COLUMN: THE BIG PICTURE ===
    with left_col:
        st.markdown("### 📝 Full Generated Discharge Summary")
        
        # Get Final Summaries
        final_view = stay_record["views"].get("final", {})
        flan_final = final_view.get("flan", "No summary.")
        med_final = final_view.get("meditron", "No summary.")
        
        tab1, tab2 = st.tabs(["FLAN-T5 (Final)", "Meditron-7B (Final)"])
        with tab1:
            st.success(flan_final)
        with tab2:
            st.info(med_final)

        # Conditional Display of Ground Truth
        if show_ground_truth:
            st.markdown("---")
            st.markdown("### 📄 Actual Discharge Note (Ground Truth)")
            discharge_text = stay_record.get("discharge_text", "")
            st.markdown(f"```\n{discharge_text}\n```")

    # === RIGHT COLUMN: DRILL-DOWN VIEW ===
    with right_col:
        st.markdown(f"### 🔍 Focused View: {view_label}")
        
        view_data = stay_record["views"].get(current_view_slug, {})
        v_flan = view_data.get("flan", "N/A")
        v_med = view_data.get("meditron", "N/A")
        
        # --- NEW TAB STRUCTURE ---
        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
            "FLAN-T5 (View)", 
            "Meditron-7B (View)", 
            "Metrics & Visuals", 
            "🛠️ Pipeline Inspector"
        ])
        
        with v_tab1:
            st.success(v_flan)
        with v_tab2:
            st.info(v_med)
            
        # --- MERGED TAB: METRICS + VISUALS ---
        with v_tab3:
            # 1. Metrics Section (LIVE CALCULATION)
            st.markdown("#### Evaluation Metrics")
            
            # Use Live Calculation
            discharge_text = stay_record.get("discharge_text", "")
            
            # Determine which text to compare
            if current_view_slug == "final":
                txt_flan = final_view.get("flan", "")
                txt_med = final_view.get("meditron", "")
            else:
                txt_flan = v_flan
                txt_med = v_med

            live_metrics = compare_summaries(txt_flan, txt_med, discharge_text)
            f_m = live_metrics["flan"]
            m_m = live_metrics["meditron"]
                
            df_m = pd.DataFrame({
                "Metric": [
                    "BERT Precision", 
                    "Embedding Similarity", 
                    "Medical Term Density",
                    "ROUGE-1 (F1)"
                ],
                "FLAN-T5": [
                    f_m.get('bert_precision', 0), 
                    f_m.get('embedding_similarity', 0), 
                    f_m.get('medical_term_density', 0),
                    f_m.get('rouge1', 0)
                ],
                "Meditron-7B": [
                    m_m.get('bert_precision', 0), 
                    m_m.get('embedding_similarity', 0), 
                    m_m.get('medical_term_density', 0),
                    m_m.get('rouge1', 0)
                ]
            })
            st.dataframe(df_m.set_index("Metric").round(3), use_container_width=True)

            st.markdown("---")
            st.markdown("#### Data Visualizations")
            
            # 2. Visualizations Section
            if current_view_slug == "admission":
                render_admission_table(raw_stay_data)
            elif current_view_slug == "dx_proc":
                render_diagnoses_table(raw_stay_data)
                render_hosp_procedures_table(raw_stay_data)
            elif current_view_slug == "labs":
                render_labs_visuals(raw_stay_data, icu_intime=icu_intime)
            elif current_view_slug == "meds":
                render_medications_visuals(raw_stay_data, icu_intime=icu_intime)
            elif current_view_slug == "measurements":
                render_measurements_visuals(raw_stay_data, icu_intime=icu_intime)
            elif current_view_slug == "outputs":
                render_outputs_visuals(raw_stay_data)
            elif current_view_slug == "procedureevents":
                render_icu_procedureevents_table(raw_stay_data)

        # --- THE INSPECTOR TAB ---
        with v_tab4:
            # Step 1: Features (Expanded by Default)
            with st.expander("Step 1: Computational Abstraction (Feature Engineering)", expanded=True):
                st.caption("This is the raw mathematical data (Trends & Units) extracted from the database.")
                feats = view_data.get("debug_features", {})
                
                # Logic to unwrap single keys for cleaner display
                data_to_show = feats
                if isinstance(feats, dict) and len(feats) == 1 and isinstance(list(feats.values())[0], list):
                    data_to_show = list(feats.values())[0]
                
                st.json(data_to_show, expanded=True)

            # Step 2: Prompt (Expanded by Default)
            with st.expander("Step 2: Prompt Construction (Linearization)", expanded=True):
                st.caption("This is the exact instruction sent to the LLM.")
                prompt_text = view_data.get("debug_prompt_flan", "No prompt data saved.")
                st.text_area("Prompt Text", prompt_text, height=400, disabled=True)

if __name__ == "__main__":
    main()