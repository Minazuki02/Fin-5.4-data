from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedValue:
    """Container for normalized answer values."""

    text: str
    numeric_candidates: tuple[float, ...]


def normalize_text(value: Any) -> str:
    """Normalize free-form text answers for string comparison."""
    text = str(value).strip().lower()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _numeric_candidates_from_string(text: str) -> tuple[float, ...]:
    """Extract numeric candidates from a string, including percent variants."""
    cleaned = text.strip().replace(",", "").replace("$", "")
    if not cleaned:
        return ()

    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", cleaned)
    if not match:
        return ()

    is_percent = cleaned.endswith("%")
    number = float(cleaned[:-1] if is_percent else cleaned)
    if is_percent:
        return (number, number / 100.0)
    return (number,)


def normalize_value(value: Any) -> NormalizedValue:
    """Normalize an answer into comparable text and numeric candidates."""
    if isinstance(value, bool):
        text = "yes" if value else "no"
        return NormalizedValue(text=text, numeric_candidates=())

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        text = str(int(number)) if number.is_integer() else str(number)
        return NormalizedValue(text=text, numeric_candidates=(number,))

    text = normalize_text(value)
    return NormalizedValue(text=text, numeric_candidates=_numeric_candidates_from_string(text))


def numerically_equal(left: Any, right: Any, *, abs_tol: float = 1e-4, rel_tol: float = 1e-4) -> bool:
    """Compare two values numerically with tolerance and percent handling."""
    left_norm = normalize_value(left)
    right_norm = normalize_value(right)

    for left_value in left_norm.numeric_candidates:
        for right_value in right_norm.numeric_candidates:
            if math.isclose(left_value, right_value, abs_tol=abs_tol, rel_tol=rel_tol):
                return True
    return False


def answers_match(predicted: Any, gold: Any) -> bool:
    """Return whether predicted and gold answers should be treated as equal."""
    if predicted is None or gold is None:
        return False

    if numerically_equal(predicted, gold):
        return True

    predicted_norm = normalize_value(predicted)
    gold_norm = normalize_value(gold)
    return bool(predicted_norm.text) and predicted_norm.text == gold_norm.text

