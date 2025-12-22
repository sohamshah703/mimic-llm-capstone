import os
import sys
import argparse

# wire imports from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features import load_all_tables_for_stay
from generator import run_generation_for_view, TOKEN_LIMITS
from eval import compare_summaries

# Define the order for Final Summary (Must match precompute_summaries.py)
FINAL_ORDER = [
    ("dx_proc", "Diagnosis and admission context"),
    ("labs", "Laboratory events during the ICU stay"),
    ("meds", "Medications and therapies during the ICU stay"),
    ("measurements", "ICU measurements and clinical course"),
    ("procedureevents", "ICU bedside procedures and interventions"),
    ("outputs", "Fluid outputs and drains"),
]

VALID_VIEWS = ["dx_proc", "labs", "meds", "measurements", "outputs", "procedureevents", "admission", "final"]

def pretty_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run FLAN-T5 and Meditron summaries for a single stay_id using the centralized generator logic."
    )
    parser.add_argument(
        "--stay_id",
        type=int,
        required=True,
        help="ICU stay_id from the 250-stay cohort.",
    )
    parser.add_argument(
        "--view_type",
        type=str,
        choices=VALID_VIEWS,
        default="final",
        help="Which view to summarise.",
    )
    args = parser.parse_args()

    stay_id = args.stay_id
    view_type = args.view_type

    pretty_header(f"RUNNING INFERENCE FOR stay_id={stay_id}, view_type={view_type}")

    # 1. Load Data
    try:
        stay_data = load_all_tables_for_stay(stay_id)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    hadm_id = stay_data["hadm_id"]
    subject_id = stay_data["subject_id"]
    discharge_text = stay_data["discharge_text"]

    print(f"subject_id : {subject_id}")
    print(f"hadm_id    : {hadm_id}")
    print(f"stay_id    : {stay_id}")
    print(f"Discharge note present: {bool(discharge_text.strip())}")

    # 2. Run Generation via generator.py
    # This ensures we use the EXACT same logic and token limits as the main pipeline.
    
    flan_summary = ""
    meditron_summary = ""

    if view_type == "final":
        pretty_header("GENERATING MULTI-SECTION FINAL SUMMARY...")
        
        flan_parts = []
        med_parts = []

        # Iterate through the defined order and generate each piece
        for v_key, header in FINAL_ORDER:
            print(f"  -> Generating {v_key}...")
            f_text, m_text = run_generation_for_view(stay_data, v_key)
            
            flan_parts.append(f"{header}:\n{f_text}")
            med_parts.append(f"{header}:\n{m_text}")
        
        flan_summary = "\n\n".join(flan_parts)
        meditron_summary = "\n\n".join(med_parts)

    else:
        # Single View
        print(f"  -> Generating {view_type} (Limit: {TOKEN_LIMITS.get(view_type)} tokens)...")
        flan_summary, meditron_summary = run_generation_for_view(stay_data, view_type)

    # 3. Output Results
    pretty_header("FLAN-T5 SUMMARY")
    print(flan_summary)
    print("\n[Length: ~{} tokens]".format(len(flan_summary.split())))

    pretty_header("MEDITRON-7B SUMMARY")
    print(meditron_summary)
    print("\n[Length: ~{} tokens]".format(len(meditron_summary.split())))

    pretty_header("ACTUAL DISCHARGE SUMMARY (Snippet)")
    if discharge_text.strip():
        print(discharge_text[:1200])
        if len(discharge_text) > 1200:
            print("\n[... truncated ...]")
    else:
        print("(No discharge summary available)")

    # 4. Metrics
    pretty_header("EVALUATION METRICS")
    metrics = compare_summaries(flan_summary, meditron_summary, discharge_text)
    fl = metrics["flan"]
    md = metrics["meditron"]

    print(f"{'Metric':<30} | {'FLAN-T5':<12} | {'Meditron-7B':<12}")
    print("-" * 60)
    print(f"{'BERTScore Precision':<30} | {fl['bert_precision']:.3f}        | {md['bert_precision']:.3f}")
    print(f"{'Embedding similarity':<30} | {fl['embedding_similarity']:.3f}        | {md['embedding_similarity']:.3f}")
    print(f"{'Avg sentence length':<30} | {fl['avg_sentence_length']:.1f}        | {md['avg_sentence_length']:.1f}")
    print(f"{'Medical term density':<30} | {fl['medical_term_density']:.3f}        | {md['medical_term_density']:.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()