#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("negative", "neutral", "positive")
ABSTAIN_PATTERNS = (
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not enough information",
    "insufficient information",
    "unknown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FinancialPhraseBank pilot badcases.")
    parser.add_argument("--input", required=True, help="Distilled JSONL file to analyze.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary and per-category badcases.")
    parser.add_argument("--tag", default="financial_phrasebank_pilot_v1")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def extract_label(text: Any) -> str:
    normalized = normalize_text(text)
    if normalized in LABELS:
        return normalized
    for label in LABELS:
        if re.search(rf"\b{label}\b", normalized):
            return label
    return ""


def is_abstain(text: Any) -> bool:
    normalized = normalize_text(text)
    return any(pattern in normalized for pattern in ABSTAIN_PATTERNS)


def reasoning_label(text: Any) -> str:
    normalized = normalize_text(text)
    for pattern in (
        r"sentiment (?:is|should be|would be|therefore is|thus is) (negative|neutral|positive)",
        r"classif(?:y|ied) as (negative|neutral|positive)",
        r"label(?: is| should be)? (negative|neutral|positive)",
        r"overall(?: sentiment)?[: ]+(negative|neutral|positive)",
    ):
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return ""


def classify_badcase(record: dict[str, Any]) -> str:
    model_answer = str(record.get("model_answer", "")).strip()
    reference_answer = str(record.get("reference_answer", "")).strip().lower()
    reasoning = str(record.get("model_reasoning", "")).strip()

    if not model_answer:
        return "parse_error"
    if is_abstain(model_answer):
        return "abstain_error"

    pred = extract_label(model_answer)
    if not pred:
        return "label_format_error"

    reasoning_pred = reasoning_label(reasoning)
    if reasoning_pred and reasoning_pred != pred:
        return "reasoning_answer_inconsistent"

    if pred == reference_answer:
        return "correct"
    if {pred, reference_answer} == {"negative", "positive"}:
        return "polarity_flip"
    if "neutral" in {pred, reference_answer}:
        return "neutral_polarity_confusion"
    return "wrong_label_selection"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    category_counts = Counter()
    confusion = defaultdict(Counter)
    badcases: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        reference = str(record.get("reference_answer", "")).strip().lower()
        pred = extract_label(record.get("model_answer", ""))
        if reference in LABELS and pred in LABELS:
            confusion[reference][pred] += 1
        category = classify_badcase(record)
        category_counts[category] += 1
        if category != "correct":
            badcases[category].append(record)

    total = len(records)
    correct = category_counts.get("correct", 0)
    summary = {
        "tag": args.tag,
        "input_path": str(input_path),
        "total_samples": total,
        "correct_count": correct,
        "strict_accuracy": (correct / total) if total else 0.0,
        "badcase_count": total - correct,
        "badcase_type_counts": dict(category_counts),
        "confusion_matrix": {gold: dict(preds) for gold, preds in confusion.items()},
    }

    (output_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for category, rows in badcases.items():
        output_path = output_dir / f"{args.tag}_{category}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
