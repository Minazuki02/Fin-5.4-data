from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseResult:
    """Container for response parse results."""

    reasoning: str
    final_answer: str
    status: str
    error: str


def strip_code_fences(text: str) -> str:
    """Remove common markdown code fences around JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> str:
    """Extract the first JSON object substring from a text blob."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def normalize_value(text: str) -> str:
    """Normalize extracted value text."""
    value = text.strip().strip(",")
    value = re.sub(r"^(?:reasoning|final_answer|answer|label)\s*[:=\-]\s*", "", value, flags=re.IGNORECASE)
    return value.strip().strip('"').strip("'").strip()


def extract_structured_field(text: str, field_names: list[str]) -> str:
    """Extract a structured field from JSON-like or plain-text output."""
    patterns = [
        rf'"?(?:{"|".join(field_names)})"?\s*[:=]\s*"?(.*?)"?(?:\n|$)',
        rf"(?im)^(?:{'|'.join(field_names)})\s*[:=]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            value = normalize_value(match.group(1))
            if value:
                return value
    return ""


def candidate_lines(text: str) -> list[str]:
    """Return non-empty lines suitable for heuristic parsing."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def candidate_sentences(text: str) -> list[str]:
    """Split text into coarse sentences for answer extraction."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def extract_from_markers(text: str) -> str:
    """Extract final answer from common answer marker phrases."""
    marker_patterns = [
        r"(?is)(?:final answer|answer|final output|output|conclusion|therefore|thus|so)\s*[:=\-]\s*(.+?)(?:\n|$)",
        r"(?is)(?:the answer is|the final answer is|this means|this gives|this yields)\s+(.+?)(?:\n|$)",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text)
        if match:
            value = normalize_value(match.group(1))
            if value:
                return value
    return ""


def extract_sentiment_answer(text: str) -> str:
    """Extract a sentiment label from response text."""
    matches = re.findall(r"\b(negative|neutral|positive)\b", text, flags=re.IGNORECASE)
    return matches[-1].lower() if matches else ""


def extract_choice_answer(text: str, options: list[Any]) -> str:
    """Extract an MCQ-style answer from response text."""
    letter_match = re.search(r"(?i)\b(?:option|choice|answer)\s*[:=\-]?\s*([A-F])\b", text)
    if letter_match:
        return letter_match.group(1).upper()

    tail = text[-600:]
    for index, option in enumerate(options):
        option_text = str(option).strip()
        if not option_text:
            continue
        letter = chr(ord("A") + index)
        if re.search(rf"(?i)\b{re.escape(letter)}\b", tail):
            return letter
        if option_text.lower() in tail.lower():
            return option_text
    return ""


def extract_yes_no(text: str) -> str:
    """Extract a yes/no answer when present."""
    matches = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return matches[-1].lower() if matches else ""


def extract_last_conclusion(text: str) -> str:
    """Use the last substantial line or sentence as the final answer."""
    for line in reversed(candidate_lines(text)):
        if re.match(r"(?i)^step\s+\d+\s*[:.-]?", line):
            continue
        if line.startswith("{") or '"reasoning"' in line.lower():
            continue
        if len(line) >= 3:
            return normalize_value(line)

    sentences = candidate_sentences(text)
    for sentence in reversed(sentences):
        if sentence.startswith("{") or '"reasoning"' in sentence.lower():
            continue
        if len(sentence) >= 3:
            return normalize_value(sentence)
    return ""


def infer_final_answer(text: str, sample: dict[str, Any] | None) -> str:
    """Infer final answer from plain-text responses using task-aware heuristics."""
    from_markers = extract_from_markers(text)
    if from_markers:
        return from_markers

    task_type = str((sample or {}).get("task_type", "")).strip().lower()
    options = list((sample or {}).get("options", []))

    if task_type == "sentiment":
        sentiment = extract_sentiment_answer(text)
        if sentiment:
            return sentiment

    if task_type == "mcq":
        choice = extract_choice_answer(text, options)
        if choice:
            return choice

    yes_no = extract_yes_no(text)
    if yes_no:
        return yes_no

    return extract_last_conclusion(text)


def is_yes_no_question(question: str) -> bool:
    """Return whether the question expects a yes/no answer."""
    return bool(re.search(r"(?i)\b(?:is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b", question.strip()))


def wants_numeric_answer(question: str) -> bool:
    """Return whether the question likely expects a numeric answer."""
    patterns = [
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bwhat (?:is|was|were|are)\b",
        r"\btotal\b",
        r"\bamount\b",
        r"\bvalue\b",
        r"\bexpense\b",
        r"\brevenue\b",
        r"\bprice\b",
        r"\brate\b",
        r"\bpercent\b",
    ]
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def extract_numeric_answer(text: str) -> str:
    """Extract the last numeric token from answer text."""
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", text)
    return matches[-1].replace("$", "").strip() if matches else ""


def source_name(sample: dict[str, Any] | None) -> str:
    """Return a lowercase source name for source-specific parsing rules."""
    return str((sample or {}).get("source", "")).strip().lower()


def max_answer_length(sample: dict[str, Any] | None) -> int:
    """Return the per-sample answer-length guardrail."""
    task_type = str((sample or {}).get("task_type", "")).strip().lower()
    source = source_name(sample)
    if source == "finance_alpaca" and task_type == "instruction":
        return 20000
    if task_type == "instruction":
        return 800
    return 500


def postprocess_final_answer(final_answer: str, sample: dict[str, Any] | None) -> str:
    """Normalize final answers to concise task-appropriate values."""
    answer = normalize_value(final_answer)
    if not answer:
        return ""

    task_type = str((sample or {}).get("task_type", "")).strip().lower()
    question = str((sample or {}).get("question", "")).strip()
    options = list((sample or {}).get("options", []))

    if task_type == "sentiment":
        sentiment = extract_sentiment_answer(answer)
        if sentiment:
            return sentiment

    if is_yes_no_question(question):
        yes_no = extract_yes_no(answer)
        if yes_no:
            return yes_no

    if task_type == "mcq":
        choice = extract_choice_answer(answer, options)
        if choice:
            return choice

    if wants_numeric_answer(question):
        numeric_matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", answer)
        numeric = extract_numeric_answer(answer)
        if numeric and len(answer) <= 48:
            return numeric
        if numeric and len(numeric_matches) == 1:
            return numeric
        if numeric:
            return answer

    return answer


def looks_valid_answer(final_answer: str, reasoning: str, raw_text: str, sample: dict[str, Any] | None = None) -> bool:
    """Reject obviously broken answers recovered from truncated JSON-like text."""
    answer = final_answer.strip()
    if not answer:
        return False
    lowered = answer.lower()
    source = source_name(sample)
    task_type = str((sample or {}).get("task_type", "")).strip().lower()
    allows_structured_artifact = source == "finance_alpaca" and task_type == "instruction"
    if (answer.startswith("{") and not allows_structured_artifact) or '"reasoning"' in lowered or '"final_answer"' in lowered:
        return False
    if len(answer) > max_answer_length(sample):
        return False
    if reasoning and answer == reasoning.strip():
        return False
    if raw_text.strip().startswith("{") and not raw_text.strip().endswith("}"):
        if ":" in answer and len(answer.split()) > 12:
            return False
    return True


def fallback_extract(text: str, sample: dict[str, Any] | None = None) -> ParseResult:
    """Fallback parse when strict JSON parsing fails."""
    structured_reasoning = extract_structured_field(text, ["reasoning"])
    structured_answer = extract_structured_field(text, ["final_answer", "answer", "label"])

    reasoning = structured_reasoning or text.strip()
    final_answer = structured_answer or infer_final_answer(text, sample)

    final_answer = postprocess_final_answer(final_answer, sample)

    if reasoning and looks_valid_answer(final_answer, reasoning, text, sample):
        return ParseResult(reasoning=reasoning, final_answer=final_answer, status="success", error="")
    return ParseResult(reasoning="", final_answer="", status="failed", error="unable_to_parse_response")


def parse_response(raw_text: str, sample: dict[str, Any] | None = None) -> ParseResult:
    """Parse teacher model output into reasoning and final_answer fields."""
    cleaned = strip_code_fences(raw_text)
    json_blob = extract_json_object(cleaned) or cleaned

    try:
        payload = json.loads(json_blob)
        reasoning = str(payload.get("reasoning", "")).strip()
        final_answer = str(payload.get("final_answer", "")).strip()
        if not reasoning:
            reasoning = cleaned
        if not final_answer:
            final_answer = infer_final_answer(cleaned, sample)
        final_answer = postprocess_final_answer(final_answer, sample)
        if reasoning and looks_valid_answer(final_answer, reasoning, cleaned, sample):
            return ParseResult(reasoning=reasoning, final_answer=final_answer, status="success", error="")
        return ParseResult(reasoning="", final_answer="", status="failed", error="json_missing_required_fields")
    except Exception as exc:  # noqa: BLE001
        fallback = fallback_extract(cleaned, sample)
        if fallback.status == "success":
            return fallback
        return ParseResult(reasoning="", final_answer="", status="failed", error=str(exc))
