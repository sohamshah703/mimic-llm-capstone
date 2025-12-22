import torch
from features import (
    build_view_dx_proc,
    build_view_labs,
    build_view_meds,
    build_view_measurements,
    build_view_admission,
    build_view_outputs,
    build_view_procedureevents,
)
from prompts import make_prompt
from models import generate_flan, generate_meditron
from eval import compare_summaries

# Exact token limits from your plan
TOKEN_LIMITS = {
    "admission": 100,
    "dx_proc": 180,
    "labs": 256,
    "meds": 256,
    "measurements": 200,
    "outputs": 200,
    "procedureevents": 180,
}

def get_features_for_view(stay_data, view_type):
    """Helper to route view_type to the correct build function."""
    if view_type == "dx_proc": return build_view_dx_proc(stay_data)
    if view_type == "labs": return build_view_labs(stay_data)
    if view_type == "meds": return build_view_meds(stay_data)
    if view_type == "measurements": return build_view_measurements(stay_data)
    if view_type == "admission": return build_view_admission(stay_data)
    if view_type == "outputs": return build_view_outputs(stay_data)
    if view_type in ("procedureevents", "procedures_icu"): return build_view_procedureevents(stay_data)
    return {}

def run_generation_for_view(stay_data, view_type):
    """
    Generates summaries for a SINGLE view.
    Returns: 
      flan_text, meditron_text, 
      features (dict), 
      flan_prompt (str), meditron_prompt (str)
    """
    # 1. Feature Engineering (Step A: The Math)
    features = get_features_for_view(stay_data, view_type)
    limit = TOKEN_LIMITS.get(view_type, 192)

    # 2. Prompt Construction (Step B: The Translation)
    # FLAN
    flan_prompt = make_prompt(view_type, features, "flan")
    # Meditron
    med_prompt = make_prompt(view_type, features, "meditron")

    # 3. Model Inference (Step C: The Generation)
    flan_text = generate_flan(flan_prompt, max_new_tokens=limit)
    med_text = generate_meditron(med_prompt, max_new_tokens=limit, temperature=0.0)

    # Return everything needed for the "Inspector"
    return flan_text, med_text, features, flan_prompt, med_prompt