"""
models.py

Model loading and text generation utilities for:
- FLAN-T5 (Seq2Seq, instruction-tuned)
- Meditron-7B (causal LM, medical domain)

Public API:
- load_flan() -> (model, tokenizer)
- load_meditron() -> (model, tokenizer)
- generate_flan(prompt, max_new_tokens=..., num_beams=...)
- generate_meditron(prompt, max_new_tokens=..., temperature=...)
"""

import os
from typing import Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# --------------------------------------------------------------------
# Model names (configurable via env vars)
# --------------------------------------------------------------------

FLAN_MODEL_NAME = os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-large")
MEDITRON_MODEL_NAME = os.environ.get("MEDITRON_MODEL_NAME", "epfl-llm/meditron-7b")

_flan_model = None
_flan_tokenizer = None
_meditron_model = None
_meditron_tokenizer = None


# --------------------------------------------------------------------
# Device and dtype helpers
# --------------------------------------------------------------------

def _use_half() -> bool:
    """Return True if we should use float16 (on GPU)."""
    return torch.cuda.is_available()


def _dtype():
    """Return the torch dtype for model weights."""
    return torch.float16 if _use_half() else torch.float32


def _pick_device(min_free_gb: float = 8.0) -> torch.device:
    """
    Pick a device for inference.

    For simplicity and robustness against meta-tensor issues:
    - Use cuda:0 if CUDA is available.
    - Otherwise fall back to CPU.

    (The original version scanned GPU memory; here we keep it
    simpler and more predictable.)
    """
    if not torch.cuda.is_available():
        print("[models] CUDA not available, using CPU.")
        return torch.device("cpu")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"[models] Using {device} for inference (total VRAM ~{total_gb:.1f} GB).")

    if total_gb < min_free_gb:
        print(
            f"[models] GPU has < {min_free_gb} GB total memory; "
            "you may see OOM if other jobs are running."
        )
    return device


# --------------------------------------------------------------------
# FLAN-T5 loading
# --------------------------------------------------------------------

def load_flan() -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Lazy-load FLAN-T5 on a suitable device.

    Key difference vs the old version: we avoid calling .to() on
    models that might live on 'meta' by loading directly with the
    correct device / device_map.
    """
    global _flan_model, _flan_tokenizer

    if _flan_model is not None and _flan_tokenizer is not None:
        return _flan_model, _flan_tokenizer

    device = _pick_device(min_free_gb=8.0)
    print(f"[models] Loading FLAN-T5 from {FLAN_MODEL_NAME} on {device}...")

    _flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)

    # Load directly with an appropriate device configuration.
    if device.type == "cuda":
        # Use device_map="auto" so HF/accelerate handles placement.
        # No manual .to(device) to avoid meta-tensor errors.
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME,
            torch_dtype=_dtype(),
            device_map="auto",
        )
    else:
        # CPU: simple load, then .to(device) (CPU) is safe.
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME,
            torch_dtype=_dtype(),
        )
        _flan_model.to(device)

    # If, for some reason, the model still has 'meta' parameters, reload without device_map.
    if any(p.device.type == "meta" for p in _flan_model.parameters()):
        print("[models] Warning: FLAN model has 'meta' tensors; reloading on CPU to avoid errors.")
        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME,
            torch_dtype=torch.float32,
        )
        _flan_model.to(torch.device("cpu"))

    _flan_model.eval()
    return _flan_model, _flan_tokenizer


# --------------------------------------------------------------------
# Meditron loading
# --------------------------------------------------------------------

def load_meditron() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Lazy-load Meditron-7B (Llama-style causal LM) on a suitable device.

    Same meta-tensor avoidance strategy as FLAN: load with appropriate
    device_map and avoid calling .to() on potentially-meta models.
    """
    global _meditron_model, _meditron_tokenizer

    if _meditron_model is not None and _meditron_tokenizer is not None:
        return _meditron_model, _meditron_tokenizer

    device = _pick_device(min_free_gb=14.0)  # Meditron-7B is larger
    print(f"[models] Loading Meditron-7B from {MEDITRON_MODEL_NAME} on {device}...")

    _meditron_tokenizer = AutoTokenizer.from_pretrained(
        MEDITRON_MODEL_NAME,
        use_fast=False,
    )

    # Important for LLaMA-style models
    if _meditron_tokenizer.pad_token_id is None:
        _meditron_tokenizer.pad_token_id = _meditron_tokenizer.eos_token_id

    if device.type == "cuda":
        _meditron_model = AutoModelForCausalLM.from_pretrained(
            MEDITRON_MODEL_NAME,
            torch_dtype=_dtype(),
            device_map="auto",
        )
    else:
        _meditron_model = AutoModelForCausalLM.from_pretrained(
            MEDITRON_MODEL_NAME,
            torch_dtype=_dtype(),
        )
        _meditron_model.to(device)

    if any(p.device.type == "meta" for p in _meditron_model.parameters()):
        print("[models] Warning: Meditron model has 'meta' tensors; reloading on CPU to avoid errors.")
        _meditron_model = AutoModelForCausalLM.from_pretrained(
            MEDITRON_MODEL_NAME,
            torch_dtype=torch.float32,
        )
        _meditron_model.to(torch.device("cpu"))

    _meditron_model.eval()
    return _meditron_model, _meditron_tokenizer


# --------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------

def generate_flan(
    prompt: str,
    max_new_tokens: int = 160,
    num_beams: int = 2,
) -> str:
    """
    Run FLAN-T5 on a prompt and return the decoded summary.

    Defaults are tuned to:
    - avoid repetition (no_repeat_ngram_size, repetition_penalty),
    - keep decoding deterministic (no sampling),
    - keep length reasonable per view.
    """
    model, tokenizer = load_flan()
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": num_beams,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 4,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


def generate_meditron(
    prompt: str,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
) -> str:
    """
    Run Meditron-7B on a prompt and return the decoded continuation.
    STRIPS THE INPUT PROMPT from the output to avoid repetition.
    """
    model, tokenizer = load_meditron()
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048, # Increased to handle larger contexts if needed
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Dynamic params based on temp
    if temperature > 0.0:
        gen_kwargs = {
            "do_sample": True,
            "temperature": max(temperature, 1e-4),
            "top_p": 0.9,
        }
    else:
        gen_kwargs = {
            "do_sample": False,
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs
        )

    # --- KEY FIX: Slice off the input tokens ---
    # Only decode the *new* tokens generated by the model
    generated_tokens = outputs[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return text.strip()