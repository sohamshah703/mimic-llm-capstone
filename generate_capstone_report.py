#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# ========== CONFIG ==========

# Project root will usually be CAPSTONE_Project
# You can override it via CLI args if needed

# Relative paths (from project root) to "main coding files"
MAIN_CODE_REL_PATHS = [
    # Scripts
    "mimic_llm/scripts/filter_diagnoses_to_cohort.py",
    "mimic_llm/scripts/filter_discharge_to_cohort.py",
    "mimic_llm/scripts/filter_icustays_to_cohort.py",
    "mimic_llm/scripts/filter_lab_tests_to_cohort.py",
    "mimic_llm/scripts/filter_measurements_to_cohort.py",
    "mimic_llm/scripts/filter_medications_to_cohort.py",
    "mimic_llm/scripts/filter_outputevents_to_cohort.py",
    "mimic_llm/scripts/filter_patients_admissions_to_cohort.py",
    "mimic_llm/scripts/filter_procedureevents_to_cohort.py",
    "mimic_llm/scripts/filter_procedures_to_cohort.py",
    "mimic_llm/scripts/inspect_pipeline.py",
    "mimic_llm/scripts/make_cohort_icu_250.py",
    "mimic_llm/scripts/make_diagnoses_clean.py",
    "mimic_llm/scripts/make_discharge_notes_clean.py",
    "mimic_llm/scripts/make_icustays_clean.py",
    "mimic_llm/scripts/make_lab_tests_clean.py",
    "mimic_llm/scripts/make_measurements_clean.py",
    "mimic_llm/scripts/make_medications_clean.py",
    "mimic_llm/scripts/make_outputevents_clean.py",
    "mimic_llm/scripts/make_patients_admissions_clean.py",
    "mimic_llm/scripts/make_procedureevents_clean.py",
    "mimic_llm/scripts/make_procedures_clean.py",
    "mimic_llm/scripts/precompute_summaries.py",
    "mimic_llm/scripts/run_single_stay_inference.py",

    # Top-level python modules
    "mimic_llm/app_streamlit.py",
    "mimic_llm/eval.py",
    "mimic_llm/features.py",
    "mimic_llm/generator.py",
    "mimic_llm/models.py",
    "mimic_llm/paths.py",
    "mimic_llm/prompts.py",
    "mimic_llm/visuals.py",
]

# Directories to ignore when traversing structure / supporting files
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
}

DEFAULT_OUTPUT_FILENAME = "capstone_project_overview.txt"


# ========== HELPERS ==========

def build_file_structure(root: Path) -> list[str]:
    """
    Returns an indented tree-like view of all files/directories under root.
    """
    lines: list[str] = []
    lines.append(f"{root.name}/")

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(root)

        # Filter ignored directories for traversal
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        # Print directory (skip root, already done)
        if rel_dir != Path("."):
            indent_level = len(rel_dir.parts)
            indent = "    " * indent_level
            lines.append(f"{indent}{rel_dir.name}/")

        # Print files
        for filename in sorted(filenames):
            rel_file = (current_dir / filename).relative_to(root)

            # Skip files that live inside ignored directories (safety)
            if any(part in IGNORE_DIRS for part in rel_file.parts):
                continue

            indent_level = len(rel_file.parts)
            indent = "    " * indent_level
            lines.append(f"{indent}{filename}")

    return lines


def collect_supporting_files(root: Path, main_code_paths: set[Path]) -> list[Path]:
    """
    Supporting files = every file under root that is NOT:
      - in IGNORE_DIRS
      - one of the main code files
    """
    supporting: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(root)

        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            file_path = current_dir / filename
            rel_path = file_path.relative_to(root)

            if any(part in IGNORE_DIRS for part in rel_path.parts):
                continue

            if rel_path in main_code_paths:
                continue

            supporting.append(file_path)

    supporting.sort(key=lambda p: str(p.relative_to(root)))
    return supporting


def collect_main_code_files(root: Path) -> list[Path]:
    """
    Resolve MAIN_CODE_REL_PATHS against the root and return only those that exist.
    """
    result: list[Path] = []
    for rel in MAIN_CODE_REL_PATHS:
        path = (root / rel).resolve()
        if path.exists() and path.is_file():
            result.append(path)
        else:
            # Still include missing ones at the end with a note
            result.append(path)
    return result


# ========== MAIN ==========

def main():
    # Decide project root
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(".").resolve()

    if not project_root.exists() or not project_root.is_dir():
        print(f"Error: project root '{project_root}' is not a valid directory.")
        sys.exit(1)

    # Decide output file
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).resolve()
    else:
        output_path = project_root / DEFAULT_OUTPUT_FILENAME

    print(f"Project root: {project_root}")
    print(f"Output file : {output_path}")

    # Prepare sets of paths
    main_code_files = collect_main_code_files(project_root)
    main_code_rel_paths = {p.relative_to(project_root) for p in main_code_files if p.exists()}
    # Note: missing files will be handled separately

    # Section 1: file structure
    structure_lines = build_file_structure(project_root)

    # Section 2: supporting files (names only)
    supporting_files = collect_supporting_files(project_root, main_code_rel_paths)

    # Write report
    with output_path.open("w", encoding="utf-8") as out:
        # ===== SECTION 1: FILE STRUCTURE =====
        out.write("========================================\n")
        out.write("SECTION 1: FILE STRUCTURE\n")
        out.write("========================================\n\n")
        for line in structure_lines:
            out.write(line + "\n")

        # ===== SECTION 2: SUPPORTING FILES (NAMES ONLY) =====
        out.write("\n\n========================================\n")
        out.write("SECTION 2: SUPPORTING FILES (NAMES ONLY)\n")
        out.write("========================================\n\n")

        if not supporting_files:
            out.write("(No supporting files found.)\n")
        else:
            for path in supporting_files:
                rel = path.relative_to(project_root)
                out.write(str(rel) + "\n")

        # ===== SECTION 3: MAIN CODING FILES (FULL CONTENT) =====
        out.write("\n\n========================================\n")
        out.write("SECTION 3: MAIN CODING FILES (FULL CONTENT)\n")
        out.write("========================================\n")

        if not main_code_files:
            out.write("\n(No main coding files found.)\n")
        else:
            for path in main_code_files:
                rel = path.relative_to(project_root) if path.exists() else None
                out.write("\n\n---------- ")
                if rel is not None:
                    out.write(str(rel))
                else:
                    out.write(str(path))
                out.write(" ----------\n\n")

                if not path.exists():
                    out.write("[Warning: file does not exist on disk]\n")
                    continue

                try:
                    with path.open("r", encoding="utf-8") as f:
                        out.write(f.read())
                except UnicodeDecodeError:
                    out.write("[Skipped: could not decode file as UTF-8]\n")

    print("Done.")


if __name__ == "__main__":
    main()