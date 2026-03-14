#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.convfinqa_matcher import match_convfinqa_record
from src.evaluation.execution_eval import gold_answer, gold_program
from src.utils.io import ensure_parent, iter_jsonl


MISMATCH_TO_CATEGORY = {
    "abstain_error": "refusal_or_abstain",
    "sign_error": "sign_error",
    "calculation_error": "wrong_numeric",
    "text_mismatch": "other_text_mismatch",
    "rounding_or_format": "rounding_or_format",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for badcase export."""
    parser = argparse.ArgumentParser(description="Export ConvFinQA badcases by mismatch category.")
    parser.add_argument("--input", required=True, help="Path to a distilled ConvFinQA JSONL file.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports"))
    parser.add_argument("--prefix", default="convfinqa_badcases")
    parser.add_argument("--max_samples", type=int, help="Only inspect the first N records.")
    parser.add_argument("--skip_empty_predictions", action="store_true", help="Ignore rows with empty model answers.")
    return parser.parse_args()


def badcase_payload(record: dict[str, Any], *, line_no: int, category: str) -> dict[str, Any]:
    """Build the exported payload for one badcase."""
    match = match_convfinqa_record(record)
    return {
        "category": category,
        "line_no": line_no,
        "id": record.get("id", ""),
        "question": record.get("question", ""),
        "gold": match.normalized_gold,
        "prediction": match.normalized_prediction,
        "reference_answer": gold_answer(record),
        "gold_program": gold_program(record),
        "reasoning": record.get("model_reasoning", ""),
        "raw_parse_status": record.get("metadata", {}).get("distill", {}).get("parse_status", ""),
        "mismatch_type": match.mismatch_type,
    }


def main() -> int:
    """Evaluate one ConvFinQA file and export mismatch buckets."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    exported: dict[str, list[dict[str, Any]]] = {}
    summary = {
        "input_path": str(input_path),
        "sample_size": 0,
        "evaluated_count": 0,
        "correct_count": 0,
        "strict_accuracy": 0.0,
        "lenient_match_count": 0,
        "empty_prediction_count": 0,
        "exported_categories": {},
        "files": {},
    }

    for line_no, record in enumerate(iter_jsonl(input_path), start=1):
        if args.max_samples is not None and line_no > args.max_samples:
            break
        summary["sample_size"] += 1

        if str(record.get("source", "")).lower() != "convfinqa":
            raise ValueError("export_convfinqa_badcases.py only supports ConvFinQA records")

        if not str(record.get("model_answer", "")).strip():
            summary["empty_prediction_count"] += 1
            if args.skip_empty_predictions:
                continue

        match = match_convfinqa_record(record)
        summary["evaluated_count"] += 1
        if match.strict_match:
            summary["correct_count"] += 1
            continue
        if match.lenient_match:
            summary["lenient_match_count"] += 1

        category = MISMATCH_TO_CATEGORY.get(match.mismatch_type, "other_text_mismatch")
        exported.setdefault(category, []).append(badcase_payload(record, line_no=line_no, category=category))

    if summary["evaluated_count"]:
        summary["strict_accuracy"] = summary["correct_count"] / summary["evaluated_count"]

    for category, rows in exported.items():
        path = output_dir / f"{args.prefix}_{category}.jsonl"
        ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["exported_categories"][category] = len(rows)
        summary["files"][category] = str(path)

    summary_path = output_dir / f"{args.prefix}_summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
