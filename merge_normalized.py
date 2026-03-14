#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from adapters.common import NORMALIZED_ROOT


REQUIRED_FIELDS = [
    "id",
    "source",
    "task_type",
    "task_domain",
    "language",
    "question",
    "context",
    "options",
    "reference_answer",
    "reasoning",
    "metadata",
]

DEFAULT_SOURCE_FILES = [
    "financeiq.jsonl",
    "finqa.jsonl",
    "convfinqa.jsonl",
    "tatqa.jsonl",
    "fincuge_instruction.jsonl",
    "finance_alpaca.jsonl",
    "financial_phrasebank.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge normalized dataset files and validate schema consistency.")
    parser.add_argument(
        "--input-dir",
        default=str(NORMALIZED_ROOT),
        help="Directory containing normalized jsonl files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(NORMALIZED_ROOT / "merged_all.jsonl"),
        help="Path to write the merged jsonl file.",
    )
    parser.add_argument(
        "--report-path",
        default=str(NORMALIZED_ROOT / "merged_all_report.json"),
        help="Path to write the validation report JSON.",
    )
    return parser.parse_args()


def validate_record(record: dict[str, Any], source_file: str, line_number: int) -> list[str]:
    issues: list[str] = []
    record_keys = list(record.keys())
    if record_keys != REQUIRED_FIELDS:
        if set(record_keys) != set(REQUIRED_FIELDS):
            issues.append("field_set_mismatch")
        else:
            issues.append("field_order_mismatch")

    string_fields = [
        "id",
        "source",
        "task_type",
        "task_domain",
        "language",
        "question",
        "context",
        "reference_answer",
        "reasoning",
    ]
    for field in string_fields:
        if not isinstance(record.get(field), str):
            issues.append(f"{field}_not_string")

    if not isinstance(record.get("options"), list):
        issues.append("options_not_list")
    if not isinstance(record.get("metadata"), dict):
        issues.append("metadata_not_dict")

    return issues


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    source_counter: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    invalid_records: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out_handle:
        for file_name in DEFAULT_SOURCE_FILES:
            input_path = input_dir / file_name
            with input_path.open("r", encoding="utf-8") as in_handle:
                for line_number, line in enumerate(in_handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    issues = validate_record(record, file_name, line_number)
                    if issues:
                        invalid_records.append(
                            {
                                "file": file_name,
                                "line_number": line_number,
                                "id": record.get("id", ""),
                                "issues": issues,
                            }
                        )

                    record_id = record.get("id", "")
                    if record_id in seen_ids:
                        duplicate_ids.append(record_id)
                    else:
                        seen_ids.add(record_id)

                    source_counter.update([record.get("source", "")])
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1

    report = {
        "merged_output_path": str(output_path),
        "source_files": DEFAULT_SOURCE_FILES,
        "required_fields": REQUIRED_FIELDS,
        "total_records": total_records,
        "source_distribution": dict(source_counter),
        "schema_consistent": len(invalid_records) == 0,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records[:100],
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:100],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["schema_consistent"] and report["duplicate_id_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
