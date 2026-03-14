#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_parent, iter_jsonl


KEEP_MISMATCH_TYPES = {"rounding", "percent_decimal_equivalent", "format_only", "none"}


def parse_args() -> argparse.Namespace:
    """Parse CLI args for filtering original records by judge output."""
    parser = argparse.ArgumentParser(description="Filter original JSONL records using lenient judge results.")
    parser.add_argument("--input", required=True, help="Original input JSONL.")
    parser.add_argument("--judged", required=True, help="Judge output JSONL.")
    parser.add_argument("--keep_output", required=True, help="Output path for kept original rows.")
    parser.add_argument("--drop_output", required=True, help="Output path for dropped original rows.")
    return parser.parse_args()


def should_keep(judged_row: dict) -> bool:
    """Return whether the judged row should be kept."""
    judge = judged_row.get("judge", {})
    if judge.get("is_consistent", False):
        return True
    return judge.get("mismatch_type", "") in KEEP_MISMATCH_TYPES


def main() -> int:
    """Split original records into keep/drop files."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    judged_path = Path(args.judged).resolve()
    keep_path = Path(args.keep_output).resolve()
    drop_path = Path(args.drop_output).resolve()

    judged_by_id = {}
    for row in iter_jsonl(judged_path):
        sample_id = str(row.get("id", ""))
        if sample_id:
            judged_by_id[sample_id] = row

    keep_count = 0
    drop_count = 0
    missing_judge_count = 0
    ensure_parent(keep_path)
    ensure_parent(drop_path)
    with keep_path.open("w", encoding="utf-8") as keep_handle, drop_path.open("w", encoding="utf-8") as drop_handle:
        for record in iter_jsonl(input_path):
            sample_id = str(record.get("id", ""))
            judged_row = judged_by_id.get(sample_id)
            if judged_row is None:
                missing_judge_count += 1
                drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                drop_count += 1
                continue

            if should_keep(judged_row):
                keep_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                keep_count += 1
            else:
                drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                drop_count += 1

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "judged_path": str(judged_path),
                "keep_output": str(keep_path),
                "drop_output": str(drop_path),
                "keep_count": keep_count,
                "drop_count": drop_count,
                "missing_judge_count": missing_judge_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
