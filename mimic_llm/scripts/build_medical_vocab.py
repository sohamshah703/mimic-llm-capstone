import os
import sys
import pandas as pd
import re

# Wire project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import HOSP_DIR, ICU_DIR

OUTPUT_VOCAB_FILE = os.path.join(PROJECT_ROOT, "medical_vocab.txt")

# Standard stopwords to exclude so we don't count "and", "the", "with" as medical terms
STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "with", "for", "on", "as", "at", "by", "from", 
    "is", "was", "are", "were", "be", "been", "has", "have", "had", "it", "that", "this",
    "or", "an", "not", "no", "yes", "other", "unspecified", "specified", "procedure",
    "patient", "history", "status", "presence", "without", "left", "right", "bilateral",
    "acute", "chronic", "disease", "disorder", "syndrome", "injury", "due", "secondary"
}
# Note: kept "acute/chronic/disease" in stopwords because while medical, 
# they are structural. If you want them counted, remove them from this list.

def extract_words(text):
    if not isinstance(text, str):
        return []
    # Only alpha characters, lowercase
    tokens = re.findall(r"[a-z]{3,}", text.lower()) # Min length 3 to avoid noise
    return [t for t in tokens if t not in STOPWORDS]

def main():
    print("Building medical vocabulary from MIMIC dictionaries...")
    vocab = set()

    # 1. Diagnoses (ICD Codes)
    path_dx = os.path.join(HOSP_DIR, "d_icd_diagnoses.csv.gz")
    if os.path.exists(path_dx):
        print(f"Reading {path_dx}...")
        df = pd.read_csv(path_dx, compression="gzip", usecols=["long_title"])
        for title in df["long_title"].dropna():
            vocab.update(extract_words(title))

    # 2. Procedures (ICD Codes)
    path_proc = os.path.join(HOSP_DIR, "d_icd_procedures.csv.gz")
    if os.path.exists(path_proc):
        print(f"Reading {path_proc}...")
        df = pd.read_csv(path_proc, compression="gzip", usecols=["long_title"])
        for title in df["long_title"].dropna():
            vocab.update(extract_words(title))

    # 3. Lab Items
    path_labs = os.path.join(HOSP_DIR, "d_labitems.csv.gz")
    if os.path.exists(path_labs):
        print(f"Reading {path_labs}...")
        df = pd.read_csv(path_labs, compression="gzip", usecols=["label"])
        for title in df["label"].dropna():
            vocab.update(extract_words(title))

    # 4. ICU Items (Meds, Vitals)
    path_items = os.path.join(ICU_DIR, "d_items.csv.gz")
    if os.path.exists(path_items):
        print(f"Reading {path_items}...")
        df = pd.read_csv(path_items, compression="gzip", usecols=["label"])
        for title in df["label"].dropna():
            vocab.update(extract_words(title))

    print(f"Total unique medical terms found: {len(vocab)}")
    
    # Save to file
    with open(OUTPUT_VOCAB_FILE, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")
    
    print(f"Saved vocabulary to: {OUTPUT_VOCAB_FILE}")

if __name__ == "__main__":
    main()