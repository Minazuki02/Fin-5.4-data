#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from adapters.common import NORMALIZED_ROOT


TRAIN_SPLITS = {"train"}
VAL_SPLITS = {"dev", "val", "eval", "validation"}
TEST_SPLITS = {"test", "test_gold"}
FINAL_TEST_SOURCES = {"finqa", "convfinqa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split normalized records into merged train/val/test files.")
    parser.add_argument(
        "--input-path",
        default=str(NORMALIZED_ROOT / "merged_all.jsonl"),
        help="Merged normalized jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(NORMALIZED_ROOT),
        help="Directory to write split jsonl files.",
    )
    parser.add_argument(
        "--report-path",
        default=str(NORMALIZED_ROOT / "split_report.json"),
        help="Path to write split report JSON.",
    )
    return parser.parse_args()


def decide_bucket(record: dict[str, Any]) -> str:
    source = str(record.get("source", "")).strip().lower()
    split = str(record.get("metadata", {}).get("split", "")).strip().lower()
    if split in TEST_SPLITS and source in FINAL_TEST_SOURCES:
        return "test"
    if split in TEST_SPLITS:
        return "val"
    if split in VAL_SPLITS:
        return "val"
    if split in TRAIN_SPLITS:
        return "train"
    return "train"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report_path).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }

    bucket_counts: Counter[str] = Counter()
    bucket_source_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    bucket_split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    with (
        input_path.open("r", encoding="utf-8") as input_handle,
        output_paths["train"].open("w", encoding="utf-8") as train_handle,
        output_paths["val"].open("w", encoding="utf-8") as val_handle,
        output_paths["test"].open("w", encoding="utf-8") as test_handle,
    ):
        handles = {
            "train": train_handle,
            "val": val_handle,
            "test": test_handle,
        }

        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            bucket = decide_bucket(record)
            handles[bucket].write(json.dumps(record, ensure_ascii=False) + "\n")

            source = str(record.get("source", ""))
            split = str(record.get("metadata", {}).get("split", "__missing__"))
            bucket_counts.update([bucket])
            bucket_source_counts[bucket].update([source])
            bucket_split_counts[bucket].update([split])

    report = {
        "input_path": str(input_path),
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "bucket_counts": dict(bucket_counts),
        "bucket_source_counts": {key: dict(value) for key, value in bucket_source_counts.items()},
        "bucket_split_counts": {key: dict(value) for key, value in bucket_split_counts.items()},
        "rules": {
            "train_splits": sorted(TRAIN_SPLITS),
            "val_splits": sorted(VAL_SPLITS),
            "test_splits": sorted(TEST_SPLITS),
            "final_test_sources": sorted(FINAL_TEST_SOURCES),
            "non_final_sources_test_splits": "val",
            "missing_or_unknown_split": "train",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
