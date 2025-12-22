import os
import sys
from typing import Any

import pandas as pd

# -------------------------------------------------------------------
# Ensure project root (/home/soham_shah/mimic_llm) is on sys.path
# -------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # one level up from scripts/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features import load_all_tables_for_stay


# Hard-coded list of stay_ids you want to export
STAY_IDS = [
    38657298,
    35527336,
    35517464,
]


TABLE_SEP = "-" * 100
STAY_SEP = "=" * 100


def safe_str(val: Any) -> str:
    """Convert values to string safely (handles timestamps, etc.)."""
    try:
        return str(val)
    except Exception:
        return repr(val)


def write_table_block(f, key: str, value: Any):
    """
    Print one table or object for a stay into the file, showing
    ALL columns and ALL rows, with clear formatting.
    """
    f.write(TABLE_SEP + "\n")
    f.write(f"TABLE: {key}\n")

    if isinstance(value, pd.DataFrame):
        n_rows, n_cols = value.shape
        f.write(f"(DataFrame with {n_rows} rows x {n_cols} columns)\n\n")

        # Show all columns explicitly
        cols = [str(c) for c in value.columns]
        f.write("COLUMNS:\n")
        f.write("  " + ", ".join(cols) + "\n\n")

        if n_rows == 0:
            f.write("[NO ROWS]\n")
        else:
            # Convert to list-of-dicts and print each row clearly
            records = value.to_dict(orient="records")
            for i, row in enumerate(records):
                f.write(f"ROW {i}:\n")
                for col in cols:
                    v = row.get(col)
                    f.write(f"  {col}: {safe_str(v)}\n")
                f.write("\n")
    else:
        f.write(f"(Non-DataFrame object of type {type(value).__name__})\n\n")
        # For non-DataFrame objects (dicts, strings, etc.), just dump repr
        f.write(safe_str(value))
        f.write("\n")

    f.write(TABLE_SEP + "\n\n")


def main():
    out_dir = os.path.join(PROJECT_ROOT, "exports")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "three_stays_actual_data.txt")

    print(f"Exporting human-readable data for {len(STAY_IDS)} stays...")
    print(f"Output file: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for stay_id in STAY_IDS:
            stay_id = int(stay_id)
            print(f"  - Processing stay_id={stay_id} ...")
            stay_data = load_all_tables_for_stay(stay_id)

            # Big header for this stay
            f.write(STAY_SEP + "\n")
            f.write(f"STAY_ID = {stay_id}\n")
            f.write(STAY_SEP + "\n\n")

            # One clearly separated block per table / key
            for key, value in stay_data.items():
                write_table_block(f, key, value)

            f.write("\n\n")

    print("Done.")
    print("You can now download or share:", out_path)


if __name__ == "__main__":
    main()