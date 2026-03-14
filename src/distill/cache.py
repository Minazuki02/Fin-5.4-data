from __future__ import annotations

from pathlib import Path

from src.utils.io import load_jsonl_by_id


def load_existing_outputs(path: Path) -> dict[str, dict]:
    """Load existing distilled outputs keyed by sample id."""
    return load_jsonl_by_id(path)


def is_filled(record: dict) -> bool:
    """Return True when both model fields are already populated."""
    return bool(str(record.get("model_answer", "")).strip()) and bool(str(record.get("model_reasoning", "")).strip())
