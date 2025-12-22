"""
eval.py

Evaluation utilities for model-generated ICU summaries.

Metrics:
- BERTScore Precision (content faithfulness vs discharge summary)
- Embedding similarity (style + content closeness)
- Average sentence length
- Medical term density (% of tokens that are medical-ish)
"""

from typing import Dict
import re

import torch
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer


# Lazy-loaded embedding model
_embed_model = None

# Simple list of common ICU / medical tokens to approximate "medical tone"
_MEDICAL_TERMS = {
    "sepsis", "septic", "pneumonia", "respiratory", "failure", "ventilation", "ventilated",
    "intubated", "vasopressor", "norepinephrine", "dopamine", "epinephrine", "shock",
    "hemodynamic", "hemodynamically", "hypotension", "hypertension", "tachycardia",
    "bradycardia", "renal", "kidney", "creatinine", "dialysis", "crrt", "cardiac",
    "myocardial", "infarction", "ischemia", "stroke", "antibiotic", "anticoagulation",
    "insulin", "sedation", "propofol", "midazolam", "fentanyl", "icu", "intensive",
    "unit", "monitoring", "lactate", "acidosis", "alkalosis", "hypoxia", "hypercapnia",
}


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # Compact, fast general-purpose embedding model on CPU only
        _embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    return _embed_model


def bert_precision(pred: str, ref: str, lang: str = "en") -> float:
    """
    Compute BERTScore Precision between pred and ref.
    """
    if not pred.strip() or not ref.strip():
        return 0.0

    P, R, F1 = bert_score(
    [pred],
    [ref],
    lang=lang,
    verbose=False,
    device="cpu",   # force CPU for BERTScore
)

    return float(P[0])


def embedding_similarity(pred: str, ref: str) -> float:
    """
    Cosine similarity between sentence embeddings of pred and ref.
    Approximates style + content closeness.
    """
    if not pred.strip() or not ref.strip():
        return 0.0

    model = _get_embed_model()
    embeddings = model.encode([pred, ref], convert_to_tensor=True)
    v1, v2 = embeddings[0], embeddings[1]
    sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()
    return float(sim)


def avg_sentence_length(text: str) -> float:
    """
    Average sentence length in tokens (whitespace-split) based on a simple
    sentence split heuristic.
    """
    if not text.strip():
        return 0.0

    # naive sentence split
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    token_counts = [len(s.split()) for s in sentences if s.split()]
    if not token_counts:
        return 0.0

    return float(sum(token_counts) / len(token_counts))


def medical_term_density(text: str) -> float:
    """
    Approximate "medical-ness": fraction of tokens that appear in a small
    domain-specific vocab.

    Returns a value between 0 and 1.
    """
    if not text.strip():
        return 0.0

    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0.0

    med_hits = sum(1 for t in tokens if t in _MEDICAL_TERMS)
    return float(med_hits / len(tokens))


def compare_summaries(
    flan_summary: str,
    meditron_summary: str,
    discharge_text: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics for FLAN and Meditron summaries against
    the actual discharge summary.

    Returns:
        {
            "flan": {...},
            "meditron": {...},
        }
    """
    if not discharge_text.strip():
        # Can't compute reference-based metrics if we have no discharge note
        return {
            "flan": {
                "bert_precision": 0.0,
                "embedding_similarity": 0.0,
                "avg_sentence_length": avg_sentence_length(flan_summary),
                "medical_term_density": medical_term_density(flan_summary),
            },
            "meditron": {
                "bert_precision": 0.0,
                "embedding_similarity": 0.0,
                "avg_sentence_length": avg_sentence_length(meditron_summary),
                "medical_term_density": medical_term_density(meditron_summary),
            },
        }

    flan_metrics = {
        "bert_precision": bert_precision(flan_summary, discharge_text),
        "embedding_similarity": embedding_similarity(flan_summary, discharge_text),
        "avg_sentence_length": avg_sentence_length(flan_summary),
        "medical_term_density": medical_term_density(flan_summary),
    }

    meditron_metrics = {
        "bert_precision": bert_precision(meditron_summary, discharge_text),
        "embedding_similarity": embedding_similarity(meditron_summary, discharge_text),
        "avg_sentence_length": avg_sentence_length(meditron_summary),
        "medical_term_density": medical_term_density(meditron_summary),
    }

    return {"flan": flan_metrics, "meditron": meditron_metrics}