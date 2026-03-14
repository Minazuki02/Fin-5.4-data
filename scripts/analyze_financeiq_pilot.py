#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_OPTIONS = {"A", "B", "C", "D"}
ABSTAIN_PATTERNS = (
    "insufficient information",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "unknown",
    "无法判断",
    "无法确定",
    "信息不足",
    "不能确定",
    "无法得出",
)
REASONING_OPTION_PATTERNS = (
    r"(?:最终答案|final_answer|答案|故选|应选|选择|选|因此选|所以选|综上选)\s*[:：]?\s*([A-D])\b",
    r"(?:正确的是|错误的是|不正确的是|不属于的是|最合理的是)\s*[:：]?\s*([A-D])\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FinanceIQ pilot distillation outputs.")
    parser.add_argument("--input", required=True, help="Distilled FinanceIQ JSONL path.")
    parser.add_argument("--output_dir", required=True, help="Directory for reports.")
    parser.add_argument("--prefix", default="financeiq_pilot")
    return parser.parse_args()


def is_abstain(text: str) -> bool:
    lowered = text.strip().lower()
    return any(pattern in lowered for pattern in ABSTAIN_PATTERNS)


def normalized_answer(text: str) -> str:
    return str(text or "").strip()


def exact_option_letter(text: str) -> str:
    answer = normalized_answer(text).upper()
    return answer if answer in VALID_OPTIONS else ""


def option_text_map(options: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        option_text = str(option or "").strip()
        if not option_text:
            continue
        mapping[option_text] = letter
        stripped = re.sub(rf"^{letter}\s*[\.\)．、:：-]?\s*", "", option_text, flags=re.IGNORECASE).strip()
        if stripped:
            mapping[stripped] = letter
    return mapping


def infer_answer_from_text(answer: str, options: list[Any]) -> tuple[str, str]:
    raw = normalized_answer(answer)
    letter = exact_option_letter(raw)
    if letter:
        return letter, "exact_letter"

    match = re.fullmatch(r"\s*([A-Da-d])\s*[\.\)．、:：-]?\s*.+", raw, flags=re.DOTALL)
    if match:
        return match.group(1).upper(), "letter_with_extra_text"

    mapping = option_text_map(options)
    if raw in mapping:
        return mapping[raw], "option_text_exact"
    return "", "unmapped"


def infer_reasoning_choice(reasoning: str, options: list[Any]) -> str:
    text = normalized_answer(reasoning)
    if not text:
        return ""

    for pattern in REASONING_OPTION_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    mapping = option_text_map(options)
    found: list[str] = []
    for option_text, letter in mapping.items():
        if len(option_text) < 2:
            continue
        if option_text in text:
            found.append(letter)
    if len(found) == 1:
        return found[0]
    return ""


def classify_badcase(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    reference = normalized_answer(record.get("reference_answer", "")).upper()
    model_answer = normalized_answer(record.get("model_answer", ""))
    reasoning = normalized_answer(record.get("model_reasoning", ""))
    options = list(record.get("options") or [])
    distill = record.get("metadata", {}).get("distill", {})
    parse_status = normalized_answer(distill.get("parse_status", ""))
    raw_response = normalized_answer(distill.get("raw_response_text", ""))

    if parse_status != "success":
        return "parse_error", {
            "parse_status": parse_status,
            "parse_error": normalized_answer(distill.get("parse_error", "")),
        }

    if is_abstain(model_answer) or is_abstain(raw_response):
        return "abstain_error", {}

    inferred_answer, answer_kind = infer_answer_from_text(model_answer, options)
    reasoning_choice = infer_reasoning_choice(reasoning, options)

    if not inferred_answer:
        if model_answer:
            return "answer_text_mismatch", {"answer_kind": answer_kind}
        return "unknown", {"answer_kind": answer_kind}

    if inferred_answer == reference and answer_kind != "exact_letter":
        return "option_format_error", {"answer_kind": answer_kind, "normalized_choice": inferred_answer}

    if reasoning_choice and reasoning_choice != inferred_answer:
        return "reasoning_answer_inconsistent", {
            "answer_kind": answer_kind,
            "normalized_choice": inferred_answer,
            "reasoning_choice": reasoning_choice,
        }

    if inferred_answer != reference:
        if answer_kind == "exact_letter":
            return "wrong_option_selection", {
                "answer_kind": answer_kind,
                "normalized_choice": inferred_answer,
            }
        return "answer_text_mismatch", {
            "answer_kind": answer_kind,
            "normalized_choice": inferred_answer,
        }

    return None


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    parse_status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    badcases_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1

            distill = record.get("metadata", {}).get("distill", {})
            parse_status = normalized_answer(distill.get("parse_status", ""))
            parse_status_counts[parse_status or "missing"] += 1

            reference = normalized_answer(record.get("reference_answer", "")).upper()
            model_answer = normalized_answer(record.get("model_answer", "")).upper()
            if parse_status == "success" and model_answer == reference:
                correct += 1
                continue

            result = classify_badcase(record)
            if result is None:
                correct += 1
                continue

            category, extra = result
            category_counts[category] += 1

            raw_sample = record.get("metadata", {}).get("raw_sample", {})
            subject = ""
            if isinstance(raw_sample, dict):
                subject = str(raw_sample.get("file_name", ""))
            subject_counts[subject or "unknown"] += 1

            payload = {
                "line_no": line_no,
                "id": record.get("id", ""),
                "question": record.get("question", ""),
                "options": record.get("options", []),
                "reference_answer": reference,
                "model_answer": normalized_answer(record.get("model_answer", "")),
                "model_reasoning": normalized_answer(record.get("model_reasoning", "")),
                "parse_status": parse_status,
                "parse_error": normalized_answer(distill.get("parse_error", "")),
                "teacher_model": normalized_answer(distill.get("teacher_model", "")),
                "prompt_version": normalized_answer(distill.get("prompt_version", "")),
                "subject": subject or "unknown",
            }
            payload.update(extra)
            badcases_by_category[category].append(payload)

    for category, rows in badcases_by_category.items():
        path = output_dir / f"{args.prefix}_{category}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_path": str(input_path),
        "total_samples": total,
        "correct_count": correct,
        "strict_accuracy": (correct / total) if total else 0.0,
        "badcase_count": total - correct,
        "category_counts": dict(category_counts),
        "parse_status_counts": dict(parse_status_counts),
        "top_subjects_with_badcases": subject_counts.most_common(10),
        "files": {
            category: str(output_dir / f"{args.prefix}_{category}.jsonl")
            for category in sorted(badcases_by_category.keys())
        },
    }

    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
