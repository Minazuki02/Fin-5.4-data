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

from src.evaluation.normalizers import answers_match, normalize_text
from src.utils.io import ensure_parent, iter_jsonl


CATEGORIES = [
    "rounding_or_format_error",
    "percentage_or_unit_error",
    "sign_error",
    "evidence_misread",
    "calculation_error",
    "reasoning_answer_inconsistent",
    "abstain_error",
    "parse_error",
    "unknown",
]
ABSTAIN_PATTERNS = (
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "insufficient information",
    "not enough information",
    "not provided",
    "unknown",
)
NUMBER_RE = re.compile(r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for TAT-QA badcase analysis."""
    parser = argparse.ArgumentParser(description="Analyze TAT-QA pilot badcases by predefined categories.")
    parser.add_argument("--input", required=True, help="Path to distilled TAT-QA jsonl.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary and category exports.")
    parser.add_argument("--prefix", default="tatqa_train_sample500")
    parser.add_argument("--max_samples", type=int, help="Only analyze first N rows.")
    parser.add_argument("--examples_per_category", type=int, default=8)
    return parser.parse_args()


def numeric_values(value: Any) -> tuple[float, ...]:
    """Extract numeric values from text, including percent and parenthesis negatives."""
    text = str(value or "").strip()
    if not text:
        return ()

    values: list[float] = []
    for raw in NUMBER_RE.findall(text):
        token = raw.strip()
        if not token:
            continue
        negative = token.startswith("(") and token.endswith(")")
        cleaned = token.strip("()").replace("$", "").replace(",", "")
        is_percent = cleaned.endswith("%")
        if is_percent:
            cleaned = cleaned[:-1]
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if negative:
            number = -abs(number)
        values.append(number)
        if is_percent:
            values.append(number / 100.0)
    return tuple(values)


def has_percent(text: Any) -> bool:
    """Return whether text explicitly indicates a percentage."""
    lowered = normalize_text(text)
    return "%" in str(text) or "percent" in lowered or "percentage" in lowered


def unit_markers(text: Any) -> set[str]:
    """Collect coarse unit markers from answer text."""
    lowered = normalize_text(text)
    units: set[str] = set()
    if re.search(r"\bthousand\b|\bk\b", lowered):
        units.add("thousand")
    if re.search(r"\bmillion\b|\bmn\b|\bmm\b", lowered):
        units.add("million")
    if re.search(r"\bbillion\b|\bbn\b", lowered):
        units.add("billion")
    return units


def close(a: float, b: float, *, rel_tol: float = 0.01, abs_tol: float = 0.5) -> bool:
    """Short helper for tolerant numeric comparison."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def lenient_numeric_match(prediction: Any, gold: Any) -> bool:
    """Allow minor rounding differences."""
    left = numeric_values(prediction)
    right = numeric_values(gold)
    for l in left:
        for r in right:
            if close(l, r):
                return True
    return False


def is_abstain(text: Any) -> bool:
    """Detect abstention/refusal answers."""
    lowered = normalize_text(text)
    return any(marker in lowered for marker in ABSTAIN_PATTERNS)


def is_sign_error(prediction: Any, gold: Any) -> bool:
    """Detect sign flips with similar magnitude."""
    left = numeric_values(prediction)
    right = numeric_values(gold)
    for l in left:
        for r in right:
            if l * r < 0 and close(abs(l), abs(r), rel_tol=0.01, abs_tol=0.5):
                return True
    return False


def is_percentage_or_unit_error(prediction: Any, gold: Any, record: dict[str, Any]) -> bool:
    """Detect percent/unit mismatches, including x100/x1000 style scaling mistakes."""
    pred_text = str(prediction or "")
    gold_text = str(gold or "")
    pred_nums = numeric_values(pred_text)
    gold_nums = numeric_values(gold_text)

    pred_pct = has_percent(pred_text)
    gold_pct = has_percent(gold_text)
    if pred_pct != gold_pct:
        if pred_nums and gold_nums:
            for p in pred_nums:
                for g in gold_nums:
                    if close(p, g) or close(p, g * 100.0) or close(p * 100.0, g):
                        return True
        return True

    pred_units = unit_markers(pred_text)
    gold_units = unit_markers(gold_text)
    if pred_units != gold_units and pred_nums and gold_nums:
        scale_candidates = (1000.0, 1000000.0, 1000000000.0)
        for p in pred_nums:
            for g in gold_nums:
                if g == 0:
                    continue
                ratio = abs(p / g)
                if any(close(ratio, scale, rel_tol=0.02, abs_tol=0.5) for scale in scale_candidates):
                    return True
                if any(close(ratio * scale, 1.0, rel_tol=0.02, abs_tol=0.5) for scale in scale_candidates):
                    return True

    scale = str(record.get("metadata", {}).get("raw_question", {}).get("scale", "")).strip().lower()
    if scale == "percent" and pred_nums and gold_nums and any(close(p, g * 100.0) or close(p * 100.0, g) for p in pred_nums for g in gold_nums):
        return True
    if scale in {"thousand", "million", "billion"} and pred_nums and gold_nums:
        multiplier = {"thousand": 1000.0, "million": 1000000.0, "billion": 1000000000.0}[scale]
        if any(close(p, g * multiplier, rel_tol=0.02, abs_tol=1.0) or close(p * multiplier, g, rel_tol=0.02, abs_tol=1.0) for p in pred_nums for g in gold_nums):
            return True
    return False


def is_rounding_or_format_error(prediction: Any, gold: Any) -> bool:
    """Detect formatting or small rounding differences."""
    pred_text = str(prediction or "").strip()
    gold_text = str(gold or "").strip()
    if not pred_text or not gold_text:
        return False

    pred_nums = numeric_values(pred_text)
    gold_nums = numeric_values(gold_text)
    if pred_nums and gold_nums:
        for p in pred_nums:
            for g in gold_nums:
                if close(p, g, rel_tol=0.02, abs_tol=1.0):
                    return True
        if len(pred_nums) == len(gold_nums) and all(
            close(p, g, rel_tol=0.02, abs_tol=1.0) for p, g in zip(pred_nums, gold_nums)
        ):
            return True

    if pred_nums and gold_nums and sorted(round(v, 6) for v in pred_nums) == sorted(round(v, 6) for v in gold_nums):
        return True

    compact_pred = re.sub(r"[\s|,.$()%\-:;]", "", pred_text.lower())
    compact_gold = re.sub(r"[\s|,.$()%\-:;]", "", gold_text.lower())
    return bool(compact_pred) and compact_pred == compact_gold


def extract_conclusion_text(reasoning: str) -> str:
    """Extract the model's own conclusion text from reasoning."""
    if not reasoning.strip():
        return ""
    matches = re.findall(r"(?im)^conclusion\s*:\s*(.+)$", reasoning)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def is_reasoning_answer_inconsistent(record: dict[str, Any]) -> bool:
    """Detect mismatch between reasoning conclusion and final_answer."""
    answer = str(record.get("model_answer", "")).strip()
    reasoning = str(record.get("model_reasoning", "")).strip()
    if not answer or not reasoning:
        return False

    conclusion = extract_conclusion_text(reasoning)
    if not conclusion:
        return False
    if answers_match(conclusion, answer):
        return False
    if lenient_numeric_match(conclusion, answer):
        return False

    answer_norm = normalize_text(answer)
    conclusion_norm = normalize_text(conclusion)
    if answer_norm and answer_norm in conclusion_norm:
        return False
    return True


def answer_type(record: dict[str, Any]) -> str:
    """Read answer_type from metadata when available."""
    raw_question = record.get("metadata", {}).get("raw_question", {})
    if isinstance(raw_question, dict):
        return str(raw_question.get("answer_type", "")).strip().lower()
    return ""


def classify_badcase(record: dict[str, Any]) -> str:
    """Classify one mismatched TAT-QA prediction into one category."""
    parse_status = str(record.get("metadata", {}).get("distill", {}).get("parse_status", ""))
    prediction = str(record.get("model_answer", "")).strip()
    gold = str(record.get("reference_answer", "")).strip()

    if parse_status != "success" or not prediction:
        return "parse_error"
    if is_abstain(prediction):
        return "abstain_error"
    if is_reasoning_answer_inconsistent(record):
        return "reasoning_answer_inconsistent"
    if is_percentage_or_unit_error(prediction, gold, record):
        return "percentage_or_unit_error"
    if is_sign_error(prediction, gold):
        return "sign_error"
    if is_rounding_or_format_error(prediction, gold):
        return "rounding_or_format_error"

    kind = answer_type(record)
    pred_nums = numeric_values(prediction)
    gold_nums = numeric_values(gold)
    if kind in {"span", "multi-span", "count"}:
        return "evidence_misread"
    if kind in {"arithmetic"} and pred_nums and gold_nums:
        return "calculation_error"
    if pred_nums and gold_nums:
        return "calculation_error"
    return "unknown"


def payload(record: dict[str, Any], *, line_no: int, category: str) -> dict[str, Any]:
    """Build one exported badcase row."""
    return {
        "line_no": line_no,
        "category": category,
        "id": record.get("id", ""),
        "question": record.get("question", ""),
        "reference_answer": record.get("reference_answer", ""),
        "model_answer": record.get("model_answer", ""),
        "model_reasoning": record.get("model_reasoning", ""),
        "answer_type": answer_type(record),
        "scale": str(record.get("metadata", {}).get("raw_question", {}).get("scale", "")),
        "parse_status": record.get("metadata", {}).get("distill", {}).get("parse_status", ""),
    }


def main() -> int:
    """Analyze TAT-QA distilled outputs and export badcase buckets."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    badcases: dict[str, list[dict[str, Any]]] = {key: [] for key in CATEGORIES}
    total = 0
    correct = 0
    parsed_ok = 0

    for line_no, record in enumerate(iter_jsonl(input_path), start=1):
        if args.max_samples is not None and line_no > args.max_samples:
            break
        if str(record.get("source", "")).lower() != "tatqa":
            raise ValueError("analyze_tatqa_badcases.py only supports source=tatqa")

        total += 1
        parse_status = str(record.get("metadata", {}).get("distill", {}).get("parse_status", ""))
        if parse_status == "success":
            parsed_ok += 1

        prediction = record.get("model_answer", "")
        gold = record.get("reference_answer", "")
        if answers_match(prediction, gold):
            correct += 1
            continue

        category = classify_badcase(record)
        if category not in badcases:
            category = "unknown"
        badcases[category].append(payload(record, line_no=line_no, category=category))

    badcase_count = total - correct
    strict_accuracy = (correct / total) if total else 0.0

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "sample_size": total,
        "parsed_success_count": parsed_ok,
        "correct_count": correct,
        "badcase_count": badcase_count,
        "strict_accuracy": strict_accuracy,
        "category_counts": {key: len(rows) for key, rows in badcases.items()},
        "files": {},
        "examples": {},
    }

    for category, rows in badcases.items():
        if not rows:
            continue
        path = output_dir / f"{args.prefix}_{category}.jsonl"
        ensure_parent(path)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["files"][category] = str(path)
        summary["examples"][category] = rows[: args.examples_per_category]

    summary_path = output_dir / f"{args.prefix}_summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
