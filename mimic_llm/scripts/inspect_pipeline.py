import os
import sys
import json
import argparse
import pandas as pd

# Wire project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features import load_all_tables_for_stay
from generator import get_features_for_view, TOKEN_LIMITS
from prompts import make_prompt
from models import generate_flan, generate_meditron

def print_separator(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stay_id", type=int, default=38657298)
    parser.add_argument("--view", type=str, default="measurements", 
                        choices=["labs", "meds", "measurements", "outputs", "dx_proc"])
    args = parser.parse_args()

    # 1. Load Raw Data
    print_separator(f"1. RAW DATA LOADING (Stay: {args.stay_id})")
    try:
        stay_data = load_all_tables_for_stay(args.stay_id)
        print("Successfully loaded raw tables (Parquet).")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Computational Feature Engineering
    print_separator(f"2. COMPUTATIONAL FEATURE ENGINEERING (View: {args.view})")
    print(f"Calling feature builder for '{args.view}'...")
    
    # This calls your build_view_* function
    features = get_features_for_view(stay_data, args.view)
    
    print("\n[Generated Feature Dictionary (First 3 items)]:")
    # We print the raw dictionary to prove Trends/Units exists
    key = list(features.keys())[0] # e.g. "measurements_summary"
    items = features[key]
    
    # Print first few items nicely
    print(json.dumps(items[:3], indent=2, default=str))
    print(f"\n... (Total {len(items)} items in list)")

    # 3. Prompt Construction
    print_separator("3. PROMPT CONSTRUCTION (Linearization + Instructions)")
    
    # FLAN Prompt
    flan_prompt = make_prompt(args.view, features, "flan")
    
    print("--- FLAN-T5 PROMPT SEEN BY MODEL ---")
    print(flan_prompt)
    print("-" * 80)

    # 4. Model Inference
    print_separator("4. MODEL INFERENCE (Simulation)")
    limit = TOKEN_LIMITS.get(args.view, 192)
    print(f"Generating with FLAN-T5 (Limit: {limit} tokens)...")
    
    output = generate_flan(flan_prompt, max_new_tokens=limit)
    
    print("\n--- FINAL GENERATED OUTPUT ---")
    print(output)
    print("-" * 80)

if __name__ == "__main__":
    main()