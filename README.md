# Automated Generation of ICU Discharge Summaries using Large Language Models

### Capstone Project | Ashoka University

This repository contains the codebase for a system designed to automate the generation of clinical discharge summaries for Intensive Care Unit (ICU) patients. By leveraging structured Electronic Health Record (EHR) data from the **MIMIC-IV** database and open-source Large Language Models (**FLAN-T5** and **Meditron-7B**), this project aims to reduce the documentation burden on clinicians while maintaining factual accuracy and safety.

---

## 📖 Project Overview

### The Problem
Reconstructing a patient's medical trajectory from fragmented hospital databases is time-consuming. ICU clinicians often spend hours manually synthesizing thousands of data points (vitals, labs, medications) into a coherent narrative.

### The Solution
We developed a modular, representation-driven pipeline that:
1.  Ingests raw structured data (CSV) and converts it into efficient Parquet formats.
2.  Applies **computational feature abstraction** to calculate clinical trends and ensure dosage safety.
3.  Uses **LLMs** to generate view-specific summaries.
4.  **Deterministically assembles** these pieces into a final, transparent Discharge Summary.

---

## ⚙️ System Architecture

Unlike "black box" approaches that feed raw tables directly to an LLM, this system uses a rigorous 3-stage pipeline to ensure explainability and reduce hallucinations.

### 1. Data Engineering & Cohort Construction
*   **ETL:** Conversion of MIMIC-IV CSVs to high-performance Parquet files.
*   **Cohort Definition:** Selection of a "Mini-MIMIC" cohort ($N=250$) based on strict inclusion criteria (e.g., presence of **exactly one** valid discharge summary per admission) to establish an unambiguous ground truth for evaluation.

### 2. Computational Feature Engineering
Before text generation, raw data undergoes mathematical processing:
*   **Temporal Trend Regression:** High-frequency vitals (e.g., Heart Rate) are analyzed using Linear Regression (`numpy.polyfit`) to mathematically derive trends (Rising/Falling/Stable) rather than relying on static averages.
*   **Unit-Safe Aggregation:** Medication dosages are filtered using a **Modal Unit Imputation** algorithm to prevent magnitude errors (e.g., ensuring `mg` is never summed with `µg`).

### 3. Modular Inference & Assembly
*   **Dynamic Token Budgeting:** Each clinical view (Labs, Meds, Admission) is allocated a specific token budget to optimize context window usage.
*   **Parallel Inference:** Two models are compared:
    *   **FLAN-T5 (Seq2Seq):** Optimized for instruction following.
    *   **Meditron-7B (Causal):** Domain-adapted for medical knowledge; includes a custom post-processing layer to strip input echoes.
*   **Deterministic Concatenation:** The final narrative is assembled by stitching together validated section summaries, ensuring traceability.

---

## 🚀 Repository Structure

```bash
mimic_llm/
├── app_streamlit.py            # The Interactive Dashboard (UI)
├── features.py                 # Core logic: Trend regression, Unit safety, View building
├── generator.py                # Inference logic: Token budgeting, Model calls
├── models.py                   # GPU loading for FLAN-T5 and Meditron
├── prompts.py                  # Instruction templates and Linearization logic
├── visuals.py                  # Altair/Streamlit charting libraries
├── eval.py                     # Metrics: BERTScore, Embedding Similarity
├── paths.py                    # Configuration for data directories
├── scripts/
│   ├── precompute_summaries.py # Batch processing script (GPU backend)
│   ├── inspect_pipeline.py     # Debug tool to trace data transformation steps
│   └── make_*.py               # ETL scripts for cleaning MIMIC data
└── README.md
```

---

## 🛠️ Prerequisites & Data Access

**⚠️ IMPORTANT: MIMIC-IV Data Usage**
This project utilizes the **MIMIC-IV v3.1** database. This data is restricted access and protected by a Data Use Agreement (DUA).
*   **We cannot provide the data files.** You must acquire credentials via [PhysioNet](https://physionet.org/).
*   Once you have access, download the `hosp`, `icu`, and `note` modules.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/mimic-llm-capstone.git
    cd mimic-llm-capstone
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `torch`, `transformers`, `streamlit`, `pandas`, `numpy`, `bert_score`, `altair`.*

3.  **Configure Paths:**
    Open `paths.py` and update the directories to point to your local MIMIC-IV storage:
    ```python
    MIMIC_IV_DIR = "/path/to/your/mimic-iv-3.1"
    PROC_DIR = "/path/to/output/processed_data"
    ```

---

## 💻 Usage

### Step 1: Data Preprocessing (ETL)
Run the cleaning scripts to generate the Parquet database:
```bash
python scripts/make_admissions_clean.py
python scripts/make_icustays_clean.py
# ... (run for all make_*.py scripts)
```

### Step 2: Cohort Construction
Generate the specific 250-patient cohort used for evaluation:
```bash
python scripts/make_cohort_icu_250.py
python scripts/filter_all_tables_to_cohort.py
```

### Step 3: Backend Inference (GPU Required)
Run the batch generation script. This loads the LLMs, processes the patients, and saves the results (including debug traces) to a JSON file.
```bash
python scripts/precompute_summaries.py
```
*Features: Resumable (skips completed), Batch control, Crash-safe serialization.*

### Step 4: Launch the Dashboard
Run the Streamlit app to explore the results interactively.
```bash
streamlit run app_streamlit.py
```

---

## 📊 Dashboard Features

The application provides a split-screen interface for detailed analysis:

*   **Left Panel (The Narrative):** Displays the fully assembled Discharge Summary alongside the Ground Truth (Doctor's Note) for direct comparison.
*   **Right Panel (The Analysis):**
    *   **Visualizations:** Time-series charts for Vitals and Labs (normalized to "Hours since Admission").
    *   **Section Summaries:** View specific outputs for Medications, Procedures, etc.
    *   **Pipeline Inspector:** A transparent "Debug View" that reveals the intermediate mathematical features (Trends/Units) and the exact prompts sent to the LLM.

---

## 📈 Evaluation Metrics

The system evaluates generated summaries using:
1.  **BERTScore (Precision):** Measures semantic factual consistency against the reference note.
2.  **Embedding Similarity:** Measures stylistic alignment using cosine similarity.
3.  **Medical Term Density:** Quantifies the usage of domain-specific jargon.

---

## Authors

*   **Soham Shah** - BSc. Computer Science and Entrepreneurial Leadership - Ashoka University
*   **Himangi Parekh** - BSc. Computer Science and Entrepreneurial Leadership - Ashoka University

## Mentors
*   **Professor Lipika Dey**
*   **Professor Mayank Agarwal**

---

**Disclaimer:** This project is for research purposes only. The generated summaries should not be used for direct clinical decision-making without physician review.