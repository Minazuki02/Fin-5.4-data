from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.evaluation.execution_eval import gold_answer
from src.postprocess.convfinqa_postprocess import (
    ConvFinQAPostprocessResult,
    is_convfinqa_abstain,
    postprocess_convfinqa_answer,
    question_expects_percent,
)


@dataclass(frozen=True)
class ConvFinQAMatchResult:
    normalized_prediction: str
    normalized_gold: str
    strict_match: bool
    lenient_match: bool
    mismatch_type: str
    prediction_postprocess: ConvFinQAPostprocessResult
    gold_postprocess: ConvFinQAPostprocessResult


def numeric_candidates(text: str, *, expects_percent: bool) -> tuple[float, ...]:
    """Create numeric candidates for exact and percent-equivalent comparisons."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ()

    is_percent = cleaned.endswith("%")
    core = cleaned[:-1] if is_percent else cleaned
    try:
        value = float(core)
    except ValueError:
        return ()

    candidates = {value}
    if is_percent:
        candidates.add(value / 100.0)
    elif expects_percent and abs(value) <= 1:
        candidates.add(value * 100.0)
    return tuple(candidates)


def strict_match(prediction: str, gold: str) -> bool:
    """Apply strict canonical matching after normalization."""
    return bool(prediction) and prediction == gold


def lenient_match(prediction: str, gold: str, *, expects_percent: bool) -> bool:
    """Apply tolerant numeric/text matching for diagnostics."""
    if strict_match(prediction, gold):
        return True

    left = numeric_candidates(prediction, expects_percent=expects_percent)
    right = numeric_candidates(gold, expects_percent=expects_percent)
    for left_value in left:
        for right_value in right:
            if math.isclose(left_value, right_value, rel_tol=0.01, abs_tol=0.005):
                return True
            if expects_percent and math.isclose(left_value, right_value, rel_tol=0.01, abs_tol=0.5):
                return True
    return False


def sign_error(prediction: str, gold: str, *, expects_percent: bool) -> bool:
    """Detect when magnitude matches but sign is flipped."""
    left = numeric_candidates(prediction, expects_percent=expects_percent)
    right = numeric_candidates(gold, expects_percent=expects_percent)
    for left_value in left:
        for right_value in right:
            if math.isclose(abs(left_value), abs(right_value), rel_tol=0.01, abs_tol=0.5) and left_value * right_value < 0:
                return True
    return False


def mismatch_type(
    *,
    prediction: str,
    gold: str,
    expects_percent: bool,
    strict_ok: bool,
    lenient_ok: bool,
) -> str:
    """Classify one mismatch for ConvFinQA diagnostics."""
    if strict_ok:
        return "match"
    if is_convfinqa_abstain(prediction):
        return "abstain_error"
    if lenient_ok:
        return "rounding_or_format"
    if sign_error(prediction, gold, expects_percent=expects_percent):
        return "sign_error"
    if numeric_candidates(prediction, expects_percent=expects_percent) and numeric_candidates(gold, expects_percent=expects_percent):
        return "calculation_error"
    return "text_mismatch"


def match_convfinqa_record(record: dict[str, Any]) -> ConvFinQAMatchResult:
    """Run ConvFinQA-specific strict and lenient matching."""
    question = str(record.get("question", ""))
    prediction_post = postprocess_convfinqa_answer(
        question=question,
        model_answer=str(record.get("model_answer", "")),
        reasoning=str(record.get("model_reasoning", "")),
    )
    gold_post = postprocess_convfinqa_answer(
        question=question,
        model_answer=str(gold_answer(record) or ""),
        reasoning="",
    )
    expects_percent = question_expects_percent(question)
    strict_ok = strict_match(prediction_post.normalized_model_answer, gold_post.normalized_model_answer)
    lenient_ok = lenient_match(
        prediction_post.normalized_model_answer,
        gold_post.normalized_model_answer,
        expects_percent=expects_percent,
    )
    return ConvFinQAMatchResult(
        normalized_prediction=prediction_post.normalized_model_answer,
        normalized_gold=gold_post.normalized_model_answer,
        strict_match=strict_ok,
        lenient_match=lenient_ok,
        mismatch_type=mismatch_type(
            prediction=prediction_post.normalized_model_answer,
            gold=gold_post.normalized_model_answer,
            expects_percent=expects_percent,
            strict_ok=strict_ok,
            lenient_ok=lenient_ok,
        ),
        prediction_postprocess=prediction_post,
        gold_postprocess=gold_post,
    )
