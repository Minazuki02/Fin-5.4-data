#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
NORMALIZED_ROOT = REPO_ROOT / "data" / "normalized"
DATASET_DIRS = [
    "financeiq",
    "finqa",
    "convfinqa",
    "tatqa",
    "fincuge_instruction",
    "finance_alpaca",
    "financial_phrasebank",
]
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-dataset train/val/test files into global jsonl files.")
    parser.add_argument(
        "--input-root",
        default=str(NORMALIZED_ROOT),
        help="Root directory containing per-dataset normalized split files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(NORMALIZED_ROOT),
        help="Root directory to write global merged split files.",
    )
    return parser.parse_args()


def augment_record(record: dict[str, Any], split_name: str) -> dict[str, Any]:
    merged = dict(record)
    merged.setdefault("model_answer", "")
    if split_name in {"train", "val"}:
        merged.setdefault("model_reasoning", "")
    else:
        merged.pop("model_reasoning", None)
    return merged


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    for split_name in SPLITS:
        output_path = output_root / f"all_{split_name}.jsonl"
        total_count = 0
        source_counts: Counter[str] = Counter()

        with output_path.open("w", encoding="utf-8") as out_handle:
            for dataset_name in DATASET_DIRS:
                input_path = input_root / dataset_name / f"{split_name}.jsonl"
                with input_path.open("r", encoding="utf-8") as in_handle:
                    for line in in_handle:
                        if not line.strip():
                            continue
                        record = augment_record(json.loads(line), split_name)
                        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_count += 1
                        source_counts.update([record["source"]])

        summary[split_name] = {
            "output_path": str(output_path),
            "count": total_count,
            "source_counts": dict(source_counts),
        }

    report_path = output_root / "all_splits_report.json"
    payload = {
        "splits": summary,
        "fields_by_split": {
            "train": ["model_reasoning", "model_answer"],
            "val": ["model_reasoning", "model_answer"],
            "test": ["model_answer"],
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "splits": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
