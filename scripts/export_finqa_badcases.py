#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.execution_eval import gold_answer, gold_program
from src.evaluation.normalizers import answers_match, normalize_text, normalize_value
from src.utils.io import ensure_parent, iter_jsonl


ABSTAIN_PATTERNS = (
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "insufficient information",
    "not enough information",
    "not provided",
    "unknown",
)
MISMATCH_TO_CATEGORY = {
    "abstain_error": "refusal_or_abstain",
    "sign_error": "sign_error",
    "calculation_error": "wrong_numeric",
    "text_mismatch": "other_text_mismatch",
    "rounding_or_format": "rounding_or_format",
}
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FinQA badcase export."""
    parser = argparse.ArgumentParser(description="Export FinQA badcases by mismatch category.")
    parser.add_argument("--input", required=True, help="Path to a distilled FinQA JSONL file.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports"))
    parser.add_argument("--prefix", default="finqa_badcases")
    parser.add_argument("--max_samples", type=int, help="Only inspect the first N records.")
    parser.add_argument("--skip_empty_predictions", action="store_true", help="Ignore rows with empty model answers.")
    return parser.parse_args()


def is_abstain(text: str) -> bool:
    """Return whether an answer text is a refusal/abstention."""
    lowered = normalize_text(text)
    return any(pattern in lowered for pattern in ABSTAIN_PATTERNS)


def numeric_candidates(value: Any) -> tuple[float, ...]:
    """Extract numeric candidates, with a regex fallback for noisy answers."""
    normalized = normalize_value(value)
    if normalized.numeric_candidates:
        return normalized.numeric_candidates

    text = normalize_text(value)
    if not text:
        return ()
    tokens = NUMERIC_TOKEN_RE.findall(text)
    results: list[float] = []
    for token in tokens:
        is_percent = token.endswith("%")
        try:
            number = float(token[:-1] if is_percent else token)
        except ValueError:
            continue
        results.append(number)
        if is_percent:
            results.append(number / 100.0)
    return tuple(results)


def lenient_match(prediction: Any, gold: Any) -> bool:
    """Allow minor numeric rounding/format differences."""
    if answers_match(prediction, gold):
        return True

    left = numeric_candidates(prediction)
    right = numeric_candidates(gold)
    for left_value in left:
        for right_value in right:
            if math.isclose(left_value, right_value, rel_tol=0.01, abs_tol=0.5):
                return True

    return normalize_text(prediction) == normalize_text(gold) and bool(normalize_text(prediction))


def sign_error(prediction: Any, gold: Any) -> bool:
    """Detect when the magnitude is close but the sign is flipped."""
    left = numeric_candidates(prediction)
    right = numeric_candidates(gold)
    for left_value in left:
        for right_value in right:
            if left_value * right_value < 0 and math.isclose(abs(left_value), abs(right_value), rel_tol=0.01, abs_tol=0.5):
                return True
    return False


def mismatch_type(prediction: Any, gold: Any) -> str:
    """Classify one FinQA mismatch."""
    if answers_match(prediction, gold):
        return "match"
    if is_abstain(str(prediction)):
        return "abstain_error"
    if lenient_match(prediction, gold):
        return "rounding_or_format"
    if sign_error(prediction, gold):
        return "sign_error"
    if numeric_candidates(prediction) and numeric_candidates(gold):
        return "calculation_error"
    return "text_mismatch"


def normalized_prediction(record: dict[str, Any]) -> str:
    """Return a normalized prediction string for export."""
    answer = str(record.get("model_answer", "")).strip()
    return normalize_text(answer) if answer else ""


def normalized_gold(record: dict[str, Any]) -> str:
    """Return a normalized gold answer string for export."""
    return normalize_text(gold_answer(record))


def badcase_payload(record: dict[str, Any], *, line_no: int, category: str, mismatch: str) -> dict[str, Any]:
    """Build the exported payload for one badcase."""
    return {
        "category": category,
        "line_no": line_no,
        "id": record.get("id", ""),
        "question": record.get("question", ""),
        "gold": normalized_gold(record),
        "prediction": normalized_prediction(record),
        "reference_answer": gold_answer(record),
        "gold_program": gold_program(record),
        "reasoning": record.get("model_reasoning", ""),
        "raw_parse_status": record.get("metadata", {}).get("distill", {}).get("parse_status", ""),
        "mismatch_type": mismatch,
    }


def main() -> int:
    """Evaluate one FinQA file and export mismatch buckets."""
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

        if str(record.get("source", "")).lower() != "finqa":
            raise ValueError("export_finqa_badcases.py only supports FinQA records")

        if not str(record.get("model_answer", "")).strip():
            summary["empty_prediction_count"] += 1
            if args.skip_empty_predictions:
                continue

        prediction = record.get("model_answer", "")
        gold = gold_answer(record)
        strict_ok = answers_match(prediction, gold)
        summary["evaluated_count"] += 1
        if strict_ok:
            summary["correct_count"] += 1
            continue

        mismatch = mismatch_type(prediction, gold)
        if mismatch == "rounding_or_format":
            summary["lenient_match_count"] += 1
        category = MISMATCH_TO_CATEGORY.get(mismatch, "other_text_mismatch")
        exported.setdefault(category, []).append(
            badcase_payload(record, line_no=line_no, category=category, mismatch=mismatch)
        )

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
