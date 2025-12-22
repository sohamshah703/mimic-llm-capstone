#!/usr/bin/env python

import os
import sys
import gzip
import pandas as pd

# Try to use pyarrow metadata for Parquet if available (much faster)
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# --- Wire up project root / paths.py ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Adjust these imports if your constant names differ
from paths import PROC_DIR, PROC_COHORT_DIR  # mimic_proc_data, mimic_proc_data_cohort


def count_csv_rows(path: str) -> int:
    """
    Count number of data rows in a CSV/CSV.GZ file (excluding header line).
    Streaming, so memory-safe but can take some time for very large files.
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        total_lines = sum(1 for _ in f)
    # subtract header line
    return max(total_lines - 1, 0)


def summarize_file(full_path: str):
    """
    Return (n_rows, n_cols, col_names) for a given data file.
    Supports .parquet, .csv, .csv.gz.
    """
    ext = os.path.splitext(full_path)[1].lower()
    # Handle .csv.gz separately
    if full_path.endswith(".csv.gz"):
        ext = ".csv.gz"

    # Parquet
    if ext == ".parquet":
        if pq is not None:
            pf = pq.ParquetFile(full_path)
            n_rows = pf.metadata.num_rows
            n_cols = pf.metadata.num_columns
            col_names = pf.schema.names
        else:
            # Fallback: load fully via pandas (may be slow for very large files)
            df = pd.read_parquet(full_path)
            n_rows, n_cols = df.shape
            col_names = list(df.columns)
        return n_rows, n_cols, col_names

    # CSV / CSV.GZ
    elif ext in (".csv", ".csv.gz"):
        # Get column names with a tiny read
        df_head = pd.read_csv(full_path, nrows=0)
        col_names = list(df_head.columns)
        n_cols = len(col_names)
        n_rows = count_csv_rows(full_path)
        return n_rows, n_cols, col_names

    else:
        raise ValueError(f"Unsupported file type for summary: {full_path}")


def summarize_root(root_dir: str, out_path: str, title: str):
    """
    Walk root_dir recursively, summarise all parquet/csv/csv.gz files, and
    write a text summary similar to mimic_structure_summary.txt.
    """
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Root directory: {root_dir}")
    lines.append("")

    # Collect files
    table_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Sort for deterministic order
        filenames = sorted(filenames)
        for fname in filenames:
            # Only data-ish files
            lower = fname.lower()
            if lower.endswith(".parquet") or lower.endswith(".csv") or lower.endswith(".csv.gz"):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(dirpath, root_dir)
                table_files.append((rel, fname, full))

    lines.append(f"Total tables (PARQUET/CSV/CSV.GZ files): {len(table_files)}")
    lines.append("")

    last_rel = None
    for rel_dir, fname, full_path in sorted(table_files):
        # Optional: show subdirectory when it changes
        if rel_dir != last_rel:
            if rel_dir == ".":
                subname = "(root)"
            else:
                subname = rel_dir
            lines.append("")
            lines.append(f"Subdirectory: {subname}")
            lines.append("-" * (len("Subdirectory: ") + len(subname)))
            lines.append("")
            last_rel = rel_dir

        lines.append(f"Table: {fname}")
        lines.append(f"Full path: {full_path}")

        try:
            n_rows, n_cols, col_names = summarize_file(full_path)
            lines.append(f"  Shape: {n_rows} rows x {n_cols} columns")
            lines.append("  Column names:")
            for col in col_names:
                lines.append(f"    - {col}")
        except Exception as e:
            # If something goes wrong, record the error but keep going
            lines.append(f"  ERROR reading file: {e}")

        lines.append("")  # blank line between tables

    # Write out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote summary to: {out_path}")


def main():
    # 1) Full processed dataset
    full_out = os.path.join(PROJECT_ROOT, "mimic_proc_structure_summary.txt")
    summarize_root(
        root_dir=PROC_DIR,
        out_path=full_out,
        title="mimic_proc_data folder overview",
    )

    # 2) Cohort (250-stay) processed dataset
    cohort_out = os.path.join(PROJECT_ROOT, "mimic_proc_cohort_structure_summary.txt")
    summarize_root(
        root_dir=PROC_COHORT_DIR,
        out_path=cohort_out,
        title="mimic_proc_data_cohort folder overview",
    )


if __name__ == "__main__":
    main()
