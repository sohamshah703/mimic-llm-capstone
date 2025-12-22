#!/usr/bin/env python3
"""
Generate a text snapshot of the project with 3 sections:

1. Overall file structure of the project — only including the main files / coding files and scripts.
2. All supporting files that help in the project set up etc but not contributing to the final output/project.
3. All the code in the coding files exactly copy pasted in an appropriate manner.

Important: All .py files under mimic_llm/archive/ are treated as supporting only.
Their code is NOT copied into Section 3.
"""

from pathlib import Path


# ---- Paths ----

# Home directory (e.g. /home/soham_shah)
HOME = Path.home()

# Assume this script lives in the project root: /home/soham_shah/mimic_llm
PROJECT_ROOT = Path(__file__).resolve().parent

# Output file inside the project root
OUTPUT_PATH = PROJECT_ROOT / "project_full_snapshot.txt"


# Supporting dirs at $HOME level (environment / setup, not core output)
HOME_SUPPORTING_DIRS = [
    HOME / ".cache",
    HOME / ".conda",
    HOME / ".config",
    HOME / ".dotnet",
    HOME / ".ipython",
    HOME / ".jupyter",
    HOME / ".local",
    HOME / ".mamba",
    HOME / ".nv",
    HOME / ".streamlit",
    HOME / ".vscode-server",
]

# Supporting dirs/files inside the project (setup, logs, archives, etc.)
PROJECT_SUPPORTING_DIRS = [
    PROJECT_ROOT / "archive",
    PROJECT_ROOT / "exports",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "__pycache__",  # may or may not exist
]

PROJECT_SUPPORTING_FILES = [
    PROJECT_ROOT / "project_snapshot.txt",
]


# ---- Helpers ----

def is_under(path: Path, parent: Path) -> bool:
    """Return True if path is somewhere under parent."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_main_code_file(path: Path) -> bool:
    """
    Return True for .py files that are part of the main pipeline / final output.

    Rules:
    - Must be under PROJECT_ROOT.
    - Must have .py extension.
    - Must NOT be inside archive/.
    - Must NOT be inside __pycache__.
    """
    if path.suffix != ".py":
        return False

    if not is_under(path, PROJECT_ROOT):
        return False

    # Exclude archive (supporting only)
    if is_under(path, PROJECT_ROOT / "archive"):
        return False

    # Exclude any __pycache__ just in case
    if "__pycache__" in path.parts:
        return False

    return True


def collect_main_code_files():
    """Find all main .py files in the project (excluding archive/)."""
    files = []
    for p in PROJECT_ROOT.rglob("*.py"):
        if is_main_code_file(p):
            files.append(p)
    # Sort by relative path for stable ordering
    return sorted(files, key=lambda p: str(p.relative_to(PROJECT_ROOT)))


def collect_supporting_items():
    """Collect supporting dirs/files (environment + project-level support)."""
    items = []

    # Home-level supporting dirs
    for p in HOME_SUPPORTING_DIRS:
        if p.exists():
            items.append(p)

    # Project-level supporting dirs/files
    for p in PROJECT_SUPPORTING_DIRS + PROJECT_SUPPORTING_FILES:
        if p.exists():
            items.append(p)

    # Also list individual files inside archive/ as supporting
    archive_dir = PROJECT_ROOT / "archive"
    if archive_dir.exists():
        for p in archive_dir.rglob("*"):
            if p.is_file():
                items.append(p)

    # De-duplicate
    seen = set()
    unique_items = []
    for p in items:
        s = str(p)
        if s not in seen:
            unique_items.append(p)
            seen.add(s)

    return sorted(unique_items, key=str)


# ---- Formatting sections ----

def format_section_1(main_files):
    """
    Section 1: Overall file structure of the project — only including
    the main files / coding files and scripts.
    """
    lines = []
    lines.append("SECTION 1: Main project file structure (core .py files and scripts)")
    lines.append("")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    for p in main_files:
        rel = p.relative_to(PROJECT_ROOT)
        # e.g. "app_streamlit.py" or "scripts/make_cohort_icu_250.py"
        lines.append(str(rel))

    lines.append("")  # trailing blank line
    return "\n".join(lines)


def format_section_2(supporting_items):
    """
    Section 2: All supporting files that help in the project set up etc
    but not contributing directly to the final output/project.
    """
    lines = []
    lines.append("SECTION 2: Supporting / setup files and directories")
    lines.append("")
    lines.append("These paths are environment/config/logging/archive/etc., not part of the final output.")
    lines.append("")

    for p in supporting_items:
        prefix = "DIR " if p.is_dir() else "FILE"
        lines.append(f"{prefix}: {p}")

    lines.append("")
    return "\n".join(lines)


def format_section_3(main_files):
    """
    Section 3: All the code in the coding files exactly copy pasted.
    Only includes main .py files (excludes archive/ by design).
    """
    lines = []
    lines.append("SECTION 3: Full source code of main project files")
    lines.append("")

    for p in main_files:
        rel = p.relative_to(PROJECT_ROOT)
        lines.append("=" * 80)
        lines.append(f"FILE: {rel}")
        lines.append("=" * 80)

        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback if there are unexpected encodings
            text = p.read_text(errors="replace")

        # Append the file contents exactly, without modification
        lines.append(text)
        lines.append("")  # blank line between files

    return "\n".join(lines)


# ---- Main ----

def main():
    if not PROJECT_ROOT.exists():
        raise SystemExit(f"Project root does not exist: {PROJECT_ROOT}")

    main_files = collect_main_code_files()
    supporting_items = collect_supporting_items()

    section1 = format_section_1(main_files)
    section2 = format_section_2(supporting_items)
    section3 = format_section_3(main_files)

    content = "\n".join([
        section1,
        "\n" + "-" * 80 + "\n",
        section2,
        "\n" + "-" * 80 + "\n",
        section3,
        "",
    ])

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote snapshot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()