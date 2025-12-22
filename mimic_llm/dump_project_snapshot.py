#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

# ---------- CONFIG ----------

PROJECT_ROOT = Path("/home/soham_shah/mimic_llm")
OUTPUT_PATH = PROJECT_ROOT / "project_snapshot.txt"

# Coding files you listed
CODING_FILE_PATHS = [
    "/home/soham_shah/mimic_llm/scripts",
    "/home/soham_shah/mimic_llm/scripts/check_cohort_discharge_consistency.py",
    "/home/soham_shah/mimic_llm/scripts/check_stayid_hadmid_consistency.py",
    "/home/soham_shah/mimic_llm/scripts/debug_prompt_single_stay.py",
    "/home/soham_shah/mimic_llm/scripts/explore_mimic_proc_stats.py",
    "/home/soham_shah/mimic_llm/scripts/explore_proc_structure.py",
    "/home/soham_shah/mimic_llm/scripts/export_three_stays_jsonl.py",
    "/home/soham_shah/mimic_llm/scripts/export_three_stays_txt.py",
    "/home/soham_shah/mimic_llm/scripts/filter_diagnoses_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_discharge_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_icustays_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_lab_tests_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_measurements_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_medications_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_outputevents_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_patients_admissions_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_procedureevents_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/filter_procedures_to_cohort.py",
    "/home/soham_shah/mimic_llm/scripts/inspect_single_stay.py",
    "/home/soham_shah/mimic_llm/scripts/make_cohort_icu_250.py",
    "/home/soham_shah/mimic_llm/scripts/make_diagnoses_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_discharge_notes_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_icustays_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_lab_tests_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_measurements_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_medications_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_outputevents_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_patients_admissions_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_procedureevents_clean.py",
    "/home/soham_shah/mimic_llm/scripts/make_procedures_clean.py",
    "/home/soham_shah/mimic_llm/scripts/run_single_stay_inference.py",
    "/home/soham_shah/mimic_llm/app_streamlit.py",
    "/home/soham_shah/mimic_llm/eval.py",
    "/home/soham_shah/mimic_llm/features.py",
    "/home/soham_shah/mimic_llm/models.py",
    "/home/soham_shah/mimic_llm/paths.py",
    "/home/soham_shah/mimic_llm/prompts.py",
    "/home/soham_shah/mimic_llm/visuals.py",
]

# Supporting paths you listed (names only, no contents will be dumped)
SUPPORTING_PATHS = [
    "/home/soham_shah/.cache",
    "/home/soham_shah/.conda",
    "/home/soham_shah/.config",
    "/home/soham_shah/.dotnet",
    "/home/soham_shah/.ipython",
    "/home/soham_shah/.jupyter",
    "/home/soham_shah/.local",
    "/home/soham_shah/.mamba",
    "/home/soham_shah/.nv",
    "/home/soham_shah/.streamlit",
    "/home/soham_shah/.vscode-server",
    "/home/soham_shah/mimic_llm",
    "/home/soham_shah/mimic_llm/__pycache__",
    "/home/soham_shah/mimic_llm/exports",
    "/home/soham_shah/mimic_llm/logs",
    "/home/soham_shah/mimic_llm/notebooks",
]

# How deep the "broad" file structure tree should go
MAX_TREE_DEPTH = 2

# ---------- HELPERS ----------

def tree_lines(root: Path, max_depth: int = 3):
    """
    Return a list of lines representing a simple directory tree
    rooted at `root`, limited to `max_depth` levels.
    """
    lines = []

    def walk(path: Path, depth: int):
        indent = "    " * depth
        # Show directory name
        if depth == 0:
            lines.append(f"{path} /")
        else:
            lines.append(f"{indent}{path.name}/")

        if depth >= max_depth:
            return

        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, FileNotFoundError) as e:
            lines.append(f"{indent}  [Error reading directory: {e}]")
            return

        for entry in entries:
            if entry.is_dir():
                walk(entry, depth + 1)
            else:
                lines.append(f"{indent}    {entry.name}")

    if root.exists():
        walk(root, 0)
    else:
        lines.append(f"[MISSING] {root}")

    return lines


def resolve_coding_files():
    """
    Take the provided coding file paths, expand the scripts directory,
    and return a sorted list of unique .py files.
    """
    paths = set()

    for p_str in CODING_FILE_PATHS:
        p = Path(p_str)
        if p.is_dir():
            # include all .py files under this directory (recursively)
            for sub in p.rglob("*.py"):
                if sub.is_file():
                    paths.add(sub.resolve())
        else:
            # single file
            if p.suffix == ".py":
                paths.add(p.resolve())

    return sorted(paths)


def write_header(f, title: str):
    f.write("\n\n")
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n\n")


# ---------- MAIN ----------

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        # Top-level metadata
        out.write("PROJECT SNAPSHOT\n")
        out.write("-" * 80 + "\n")
        out.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"Project root: {PROJECT_ROOT}\n")
        out.write("-" * 80 + "\n")

        # 1) Broad file structure
        write_header(out, "1. BROAD FILE STRUCTURE (limited depth)")
        out.write(f"Root: {PROJECT_ROOT}\n\n")
        for line in tree_lines(PROJECT_ROOT, max_depth=MAX_TREE_DEPTH):
            out.write(line + "\n")

        # 2) Code contents
        write_header(out, "2. CODE FILE CONTENTS")
        coding_files = resolve_coding_files()
        if not coding_files:
            out.write("[No coding files found based on the provided paths]\n")
        else:
            for path in coding_files:
                out.write(f"\n----- BEGIN FILE: {path} -----\n")
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f_in:
                        for line in f_in:
                            out.write(line)
                except Exception as e:
                    out.write(f"[Error reading file {path}: {e}]\n")
                out.write(f"\n----- END FILE: {path} -----\n")

        # 3) Supporting paths (names only)
        write_header(out, "3. SUPPORTING FILES AND FOLDERS (names only)")
        out.write("The following supporting paths were provided. Only their paths are listed;\n")
        out.write("no contents are included.\n\n")

        for p_str in SUPPORTING_PATHS:
            p = Path(p_str)
            status = ""
            if not p.exists():
                status = " [MISSING]"
            elif p.is_dir():
                status = " [DIR]"
            elif p.is_file():
                status = " [FILE]"

            out.write(f"{p}{status}\n")

    print(f"Done. Snapshot written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()