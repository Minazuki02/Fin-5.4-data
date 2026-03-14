from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.executors import execute_program, looks_like_program
from src.evaluation.normalizers import answers_match


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _turn_index_from_id(sample_id: str) -> int | None:
    """Extract the turn index suffix used by ConvFinQA records."""
    match = re.search(r"_(\d+)$", sample_id)
    return int(match.group(1)) if match else None


def _convfinqa_turn(record: dict[str, Any]) -> dict[str, Any]:
    """Resolve the active ConvFinQA turn from metadata."""
    raw_sample = record.get("metadata", {}).get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return {}

    qa = raw_sample.get("qa")
    if isinstance(qa, dict):
        return qa

    turn_index = _turn_index_from_id(str(record.get("id", "")))
    if turn_index is not None:
        turn = raw_sample.get(f"qa_{turn_index}")
        if isinstance(turn, dict):
            return turn

    turn_keys = sorted(
        (key for key in raw_sample.keys() if re.fullmatch(r"qa_\d+", key)),
        key=lambda name: int(name.split("_")[1]),
    )
    if turn_keys:
        candidate = raw_sample.get(turn_keys[-1])
        if isinstance(candidate, dict):
            return candidate
    return {}


def gold_answer(record: dict[str, Any]) -> Any:
    """Extract the gold answer with dataset-specific fallbacks."""
    reference = record.get("reference_answer")
    if str(reference).strip():
        return reference

    source = str(record.get("source", "")).lower()
    raw_sample = record.get("metadata", {}).get("raw_sample", {})

    if source == "finqa" and isinstance(raw_sample, dict):
        qa = raw_sample.get("qa", {})
        if isinstance(qa, dict):
            return qa.get("answer") or qa.get("exe_ans")

    if source == "convfinqa":
        turn = _convfinqa_turn(record)
        return turn.get("answer") or turn.get("exe_ans")

    return reference


def gold_program(record: dict[str, Any]) -> str:
    """Extract the gold program if it is present."""
    source = str(record.get("source", "")).lower()
    raw_sample = record.get("metadata", {}).get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return ""

    if source == "finqa":
        qa = raw_sample.get("qa", {})
        if isinstance(qa, dict):
            return str(qa.get("program_re") or qa.get("program") or "")

    if source == "convfinqa":
        turn = _convfinqa_turn(record)
        return str(turn.get("program_re") or turn.get("program") or "")

    return ""


def resolved_prediction(record: dict[str, Any]) -> tuple[Any, bool]:
    """Resolve a model prediction into a comparable answer value."""
    prediction = str(record.get("model_answer", "")).strip()
    if not prediction:
        return "", False

    if looks_like_program(prediction):
        result = execute_program(prediction)
        return result.value, result.executable

    return prediction, False


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one record for execution accuracy."""
    prediction, executable = resolved_prediction(record)
    gold = gold_answer(record)
    correct = answers_match(prediction, gold)
    return {
        "id": record.get("id", ""),
        "source": record.get("source", ""),
        "correct": correct,
        "executable": executable,
        "prediction": prediction,
        "gold": gold,
        "gold_program": gold_program(record),
    }


def evaluate_file(path: Path) -> dict[str, Any]:
    """Evaluate one FinQA or ConvFinQA prediction file."""
    records = iter_jsonl(path)
    if not records:
        return {
            "execution_accuracy": 0.0,
            "executable_rate": 0.0,
            "total_samples": 0,
        }

    evaluated = [evaluate_record(record) for record in records]
    total = len(evaluated)
    correct = sum(1 for item in evaluated if item["correct"])
    executable = sum(1 for item in evaluated if item["executable"])

    return {
        "execution_accuracy": correct / total,
        "executable_rate": executable / total,
        "total_samples": total,
    }
