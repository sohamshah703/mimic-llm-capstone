import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date, datetime

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import COHORT_META_DIR
from features import load_all_tables_for_stay
from generator import run_generation_for_view
from eval import compare_summaries

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "precomputed_cohort_summaries.json")

# Defines the specific order for the Final Summary concatenation
FINAL_ORDER = [
    ("dx_proc", "Diagnosis and admission context"),
    ("labs", "Laboratory events during the ICU stay"),
    ("meds", "Medications and therapies during the ICU stay"),
    ("measurements", "ICU measurements and clinical course"),
    ("procedureevents", "ICU bedside procedures and interventions"),
    ("outputs", "Fluid outputs and drains"),
]

ALL_VIEWS = ["admission", "dx_proc", "labs", "meds", "measurements", "outputs", "procedureevents"]

class CustomJSONEncoder(json.JSONEncoder):
    """Prevent JSON crash on NumPy types or Dates."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, (date, datetime, pd.Timestamp)):
            return str(obj)
        return super().default(obj)

def load_cohort():
    path = os.path.join(COHORT_META_DIR, "cohort_icu_250.parquet")
    df = pd.read_parquet(path)
    return sorted(df["stay_id"].unique())

def main():
    stay_ids = load_cohort()
    print(f"--- PRECOMPUTE STARTED ---")
    print(f"Total Cohort Size: {len(stay_ids)}")
    print(f"Output File: {OUTPUT_FILE}")

    # 1. LOAD EXISTING PROGRESS
    all_data = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Found existing file with {len(all_data)} records. Resuming...")
        except json.JSONDecodeError:
            print("Existing file is corrupt or empty. Starting from scratch.")
            all_data = {}

    # 2. FILTER LIST (Resume Logic)
    completed_ids = set(all_data.keys())
    # Ensure strict string/int comparison match
    ids_to_process = [s for s in stay_ids if str(s) not in completed_ids]
    
    print(f"Remaining to process: {len(ids_to_process)}")

    if not ids_to_process:
        print("All stays are already processed! Exiting.")
        return

    # --- BATCH CONTROL ---
    BATCH_SIZE = 50
    current_batch = ids_to_process[:BATCH_SIZE]
    print(f"Processing next batch of {len(current_batch)} stays...")
    print("-" * 40)

    # 3. RUN LOOP (Only for current_batch)
    for stay_id in tqdm(current_batch, desc="Batch Progress"):
        stay_id = int(stay_id)
        
        try:
            stay_data = load_all_tables_for_stay(stay_id)
        except Exception as e:
            print(f"Skipping {stay_id}: Load error - {e}")
            continue

        discharge_text = stay_data["discharge_text"]
        
        entry = {
            "stay_id": stay_id,
            "subject_id": int(stay_data["subject_id"]),
            "hadm_id": int(stay_data["hadm_id"]),
            "discharge_text": discharge_text,
            "views": {}
        }

        flan_parts = {}
        med_parts = {}

        # Generate All Views
        for view in ALL_VIEWS:
            # Unpack the 5 values (includes debug info)
            f_text, m_text, feat_dict, f_prompt, m_prompt = run_generation_for_view(stay_data, view)
            
            # Compute metrics immediately
            metrics = compare_summaries(f_text, m_text, discharge_text)
            
            entry["views"][view] = {
                "flan": f_text,
                "meditron": m_text,
                "metrics": metrics,
                # SAVE INSPECTION DATA
                "debug_features": feat_dict,
                "debug_prompt_flan": f_prompt,
                "debug_prompt_meditron": m_prompt
            }
            
            flan_parts[view] = f_text
            med_parts[view] = m_text

        # Assemble Final Summary (Deterministic Concatenation)
        def assemble_final(parts_dict):
            blocks = []
            for view_key, header in FINAL_ORDER:
                text = parts_dict.get(view_key, "")
                if text:
                    blocks.append(f"{header}:\n{text}")
            return "\n\n".join(blocks)

        flan_final = assemble_final(flan_parts)
        med_final = assemble_final(med_parts)
        metrics_final = compare_summaries(flan_final, med_final, discharge_text)

        entry["views"]["final"] = {
            "flan": flan_final,
            "meditron": med_final,
            "metrics": metrics_final
        }

        all_data[str(stay_id)] = entry

        # Save intermittently (every 5 records)
        if len(all_data) % 5 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, cls=CustomJSONEncoder)

    # Final Save for this batch
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"--- BATCH COMPLETED ---")
    print(f"Total processed so far: {len(all_data)} / {len(stay_ids)}")
    if len(all_data) < len(stay_ids):
        print("Run the script again to process the next batch.")

if __name__ == "__main__":
    main()