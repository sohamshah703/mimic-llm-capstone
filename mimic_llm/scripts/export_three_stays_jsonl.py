import os
import sys
import json
from typing import Any, Dict

import pandas as pd

# -------------------------------------------------------------------
# Make sure the project root (/home/soham_shah/mimic_llm) is on sys.path
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


def df_to_records(obj: Any):
    """Convert a DataFrame to a list of dicts; leave other types unchanged."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def build_export_record(stay_id: int) -> Dict[str, Any]:
    """
    Load all cohort tables for a single stay_id and package them into
    one JSON-serialisable dict.
    """
    stay_data = load_all_tables_for_stay(stay_id)

    record: Dict[str, Any] = {"stay_id": int(stay_id)}

    # Copy everything from stay_data, converting DataFrames to plain records
    for key, value in stay_data.items():
        record[key] = df_to_records(value)

    return record


def main():
    out_dir = os.path.join(PROJECT_ROOT, "exports")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "three_stays_actual_data.jsonl")

    print(f"Exporting stay data for {len(STAY_IDS)} stays...")
    print(f"Output file: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for stay_id in STAY_IDS:
            print(f"  - Processing stay_id={stay_id} ...")
            record = build_export_record(int(stay_id))
            # default=str handles timestamps and other non-JSON-native types
            line = json.dumps(record, default=str)
            f.write(line + "\n")

    print("Done.")
    print("You can now download or share:", out_path)


if __name__ == "__main__":
    main()