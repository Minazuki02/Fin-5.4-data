#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.distill.provider import OpenAICompatibleProvider
from src.evaluation.normalizers import normalize_text
from src.utils.io import ensure_parent, iter_jsonl


ANSWER_PROMPT_PATH = REPO_ROOT / "prompts" / "answer_binary_judge_prompt.txt"
REASONING_PROMPT_PATH = REPO_ROOT / "prompts" / "reasoning_quality_judge_prompt.txt"

YES_SET = {"yes", "true", "y", "1", "correct", "是", "对", "正确"}
NO_SET = {"no", "false", "n", "0", "incorrect", "否", "错", "错误"}
JUDGMENT_HINTS = (
    "true or false",
    "yes or no",
    "是否",
    "判断",
)


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    """Parse CLI args for two-stage quality filtering."""
    parser = argparse.ArgumentParser(description="Run two-stage answer/reasoning scoring and filtering.")
    parser.add_argument("--input", required=True, help="Distilled JSONL input path.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports" / "two_stage_filter"))
    parser.add_argument("--train_data_root", default=str(REPO_ROOT / "train_data"))
    parser.add_argument("--source", help="Override source name (defaults to first row source).")
    parser.add_argument("--split", help="Override split name (defaults to first row split).")
    parser.add_argument("--max_samples", type=int, help="Limit first N samples.")
    parser.add_argument("--max_concurrency", type=int, help="Override API concurrency.")
    parser.add_argument("--answer_prompt", default=str(ANSWER_PROMPT_PATH))
    parser.add_argument("--reasoning_prompt", default=str(REASONING_PROMPT_PATH))
    parser.add_argument("--temperature", type=float, default=0.0, help="Override judging temperature.")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Override judging max tokens.")
    parser.add_argument("--include_context", action="store_true", help="Include context in answer judge prompt.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from a text blob."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def truncate_text(text: Any, limit: int) -> str:
    """Truncate text fields for bounded prompts."""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 15] + "\n[TRUNCATED]"


@lru_cache(maxsize=4)
def load_prompt(path_str: str) -> str:
    """Load and cache prompt template."""
    return Path(path_str).read_text(encoding="utf-8")


def canonical_bool(text: Any) -> str:
    """Map free-form text into yes/no where possible."""
    lowered = normalize_text(text)
    if lowered in YES_SET:
        return "yes"
    if lowered in NO_SET:
        return "no"
    return ""


def is_judgment_question(record: dict[str, Any]) -> bool:
    """Detect whether a record behaves like yes/no or true/false judgment."""
    question = str(record.get("question", "")).lower()
    reference = str(record.get("reference_answer", "")).strip()
    if any(hint in question for hint in JUDGMENT_HINTS):
        return True
    return bool(canonical_bool(reference))


def normalize_choice_text(text: Any) -> str:
    """Normalize an option-like text for robust matching."""
    lowered = normalize_text(text)
    lowered = re.sub(r"^(?:option|choice|answer)\s*", "", lowered)
    lowered = re.sub(r"^[\(\[]?([a-f])[\)\].:\-]\s*", "", lowered)
    return lowered.strip()


def parse_choice_index(answer: Any, options: list[Any]) -> int | None:
    """Parse an answer into option index if possible."""
    text = str(answer or "").strip()
    if not text:
        return None

    match = re.search(r"(?i)\b(?:option|choice|answer)?\s*[:=\-]?\s*([A-F])\b", text)
    if match:
        idx = ord(match.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx

    if re.fullmatch(r"\d+", text):
        idx = int(text) - 1
        if 0 <= idx < len(options):
            return idx

    normalized = normalize_choice_text(text)
    for idx, option in enumerate(options):
        option_norm = normalize_choice_text(option)
        if normalized and normalized == option_norm:
            return idx
        if option_norm and option_norm in normalized and len(option_norm) >= 4:
            return idx
    return None


def rule_answer_score(record: dict[str, Any]) -> tuple[int | None, str]:
    """
    Return rule score for MCQ/judgment tasks.
    Returns (score, method_detail), where score None means unresolved.
    """
    options = list(record.get("options") or [])
    reference = record.get("reference_answer", "")
    prediction = record.get("model_answer", "")

    if options:
        pred_idx = parse_choice_index(prediction, options)
        ref_idx = parse_choice_index(reference, options)
        if pred_idx is None or ref_idx is None:
            return None, "rule_mcq_unresolved"
        return (1 if pred_idx == ref_idx else 0), "rule_mcq"

    if is_judgment_question(record):
        pred_bool = canonical_bool(prediction)
        ref_bool = canonical_bool(reference)
        if not pred_bool or not ref_bool:
            return None, "rule_judgment_unresolved"
        return (1 if pred_bool == ref_bool else 0), "rule_judgment"

    return None, "rule_not_applicable"


def render_answer_prompt(template: str, record: dict[str, Any], *, include_context: bool) -> str:
    """Render answer judge prompt for one record."""
    return template.format(
        task_domain=truncate_text(record.get("task_domain", ""), 200),
        question=truncate_text(record.get("question", ""), 2500) or "N/A",
        reference_answer=truncate_text(record.get("reference_answer", ""), 1200) or "N/A",
        model_answer=truncate_text(record.get("model_answer", ""), 1200) or "N/A",
        context=truncate_text(record.get("context", ""), 4000) if include_context else "N/A",
    )


def render_reasoning_prompt(template: str, record: dict[str, Any]) -> str:
    """Render reasoning quality prompt for one record."""
    return template.format(
        task_domain=truncate_text(record.get("task_domain", ""), 200),
        question=truncate_text(record.get("question", ""), 2500) or "N/A",
        reference_answer=truncate_text(record.get("reference_answer", ""), 1200) or "N/A",
        model_answer=truncate_text(record.get("model_answer", ""), 1200) or "N/A",
        model_reasoning=truncate_text(record.get("model_reasoning", ""), 5000) or "N/A",
    )


def parse_answer_payload(text: str) -> dict[str, Any]:
    """Parse answer judge JSON payload."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    score = int(payload.get("answer_score", 0))
    return {
        "answer_score": 1 if score == 1 else 0,
        "judgment": str(payload.get("judgment", "")),
        "reason": str(payload.get("reason", "")),
    }


def parse_reasoning_payload(text: str) -> dict[str, Any]:
    """Parse reasoning judge JSON payload."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    score = int(payload.get("reasoning_score", 0))
    result = {
        "reasoning_score": 1 if score == 1 else 0,
        "overall": str(payload.get("overall", "")),
        "reason": str(payload.get("reason", "")),
    }
    for key in (
        "internal_consistency",
        "terminology_overlap",
        "step_count_ok",
        "logical_consistency",
        "content_diversity",
        "domain_relevance",
        "instruction_alignment",
    ):
        value = payload.get(key, 0)
        try:
            result[key] = 1 if int(value) == 1 else 0
        except Exception:  # noqa: BLE001
            result[key] = 0
    return result


def merge_quality_meta(record: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Attach quality filtering metadata to one record copy."""
    row = dict(record)
    metadata = dict(row.get("metadata", {}))
    quality_meta = dict(metadata.get("quality_filter", {}))
    quality_meta.update(updates)
    metadata["quality_filter"] = quality_meta
    row["metadata"] = metadata
    return row


def judge_answer_with_model(
    provider: OpenAICompatibleProvider,
    prompt_template: str,
    record: dict[str, Any],
    *,
    include_context: bool,
) -> JudgeResult:
    """Judge answer correctness with model."""
    prompt = render_answer_prompt(prompt_template, record, include_context=include_context)
    provider_result = provider.generate(prompt)
    if not provider_result.success:
        return JudgeResult(False, {}, provider_result.error, provider_result.latency_ms)
    try:
        payload = parse_answer_payload(provider_result.text)
        return JudgeResult(True, payload, "", provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(False, {}, f"parse_error: {exc}", provider_result.latency_ms)


def judge_reasoning_with_model(
    provider: OpenAICompatibleProvider,
    prompt_template: str,
    record: dict[str, Any],
) -> JudgeResult:
    """Judge reasoning quality with model."""
    prompt = render_reasoning_prompt(prompt_template, record)
    provider_result = provider.generate(prompt)
    if not provider_result.success:
        return JudgeResult(False, {}, provider_result.error, provider_result.latency_ms)
    try:
        payload = parse_reasoning_payload(provider_result.text)
        return JudgeResult(True, payload, "", provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(False, {}, f"parse_error: {exc}", provider_result.latency_ms)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to jsonl."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summary_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Build score summary for one binary key."""
    total = len(rows)
    ones = sum(1 for row in rows if int(row.get("metadata", {}).get("quality_filter", {}).get(key, 0)) == 1)
    return {
        "total": total,
        "score_1": ones,
        "score_0": total - ones,
        "rate_1": (ones / total) if total else 0.0,
    }


def main() -> int:
    """Run two-stage filtering and write outputs."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    train_data_root = Path(args.train_data_root).resolve()
    answer_template = load_prompt(str(Path(args.answer_prompt).resolve()))
    reasoning_template = load_prompt(str(Path(args.reasoning_prompt).resolve()))

    records = list(iter_jsonl(input_path))
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError(f"input has no records: {input_path}")

    source = args.source or str(records[0].get("source", "")).strip() or "unknown_source"
    split = args.split or str(records[0].get("split", "")).strip() or "unknown_split"

    config = load_config(Path(args.config).resolve())
    config["generation"]["temperature"] = float(args.temperature)
    config["generation"]["max_tokens"] = int(args.max_tokens)
    if args.max_concurrency is not None:
        config["api"]["max_concurrency"] = int(args.max_concurrency)
    max_workers = int(config["api"]["max_concurrency"])
    provider = OpenAICompatibleProvider.from_config(config)

    run_dir = output_root / source / split
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: answer correctness scoring.
    stage1_rows: list[dict[str, Any]] = [None] * len(records)  # type: ignore[list-item]
    unresolved_indices: list[int] = []
    rule_kept = 0
    rule_dropped = 0
    for idx, record in enumerate(records):
        score, detail = rule_answer_score(record)
        if score is None:
            unresolved_indices.append(idx)
            continue
        updated = merge_quality_meta(
            record,
            {
                "answer_score": int(score),
                "answer_score_method": detail,
                "answer_judge_success": True,
                "answer_judge_error": "",
                "answer_judge_latency_ms": 0,
                "answer_judge_reason": "rule_based",
            },
        )
        stage1_rows[idx] = updated
        if score == 1:
            rule_kept += 1
        else:
            rule_dropped += 1

    answer_latencies: list[int] = []
    answer_gpt_failures = 0
    if unresolved_indices:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    judge_answer_with_model,
                    provider,
                    answer_template,
                    records[idx],
                    include_context=args.include_context,
                ): idx
                for idx in unresolved_indices
            }
            for future in as_completed(futures):
                idx = futures[future]
                record = records[idx]
                result = future.result()
                answer_latencies.append(result.latency_ms)
                if result.success:
                    score = int(result.payload.get("answer_score", 0))
                    reason = str(result.payload.get("reason", ""))
                    updated = merge_quality_meta(
                        record,
                        {
                            "answer_score": score,
                            "answer_score_method": "gpt",
                            "answer_judge_success": True,
                            "answer_judge_error": "",
                            "answer_judge_latency_ms": result.latency_ms,
                            "answer_judge_reason": reason,
                            "answer_judge_payload": result.payload,
                        },
                    )
                else:
                    answer_gpt_failures += 1
                    updated = merge_quality_meta(
                        record,
                        {
                            "answer_score": 0,
                            "answer_score_method": "gpt_failed",
                            "answer_judge_success": False,
                            "answer_judge_error": result.error,
                            "answer_judge_latency_ms": result.latency_ms,
                            "answer_judge_reason": "",
                        },
                    )
                stage1_rows[idx] = updated

    answer_keep = [row for row in stage1_rows if int(row.get("metadata", {}).get("quality_filter", {}).get("answer_score", 0)) == 1]
    answer_drop = [row for row in stage1_rows if int(row.get("metadata", {}).get("quality_filter", {}).get("answer_score", 0)) == 0]

    stage1_scored_path = run_dir / f"{split}.stage1_answer_scored.jsonl"
    stage1_keep_path = run_dir / f"{split}.stage1_answer_keep.jsonl"
    stage1_drop_path = run_dir / f"{split}.stage1_answer_drop.jsonl"
    write_jsonl(stage1_scored_path, stage1_rows)
    write_jsonl(stage1_keep_path, answer_keep)
    write_jsonl(stage1_drop_path, answer_drop)

    # Stage 2: reasoning quality scoring on stage1 kept rows.
    stage2_rows: list[dict[str, Any]] = [None] * len(answer_keep)  # type: ignore[list-item]
    reasoning_latencies: list[int] = []
    reasoning_gpt_failures = 0
    if answer_keep:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    judge_reasoning_with_model,
                    provider,
                    reasoning_template,
                    record,
                ): idx
                for idx, record in enumerate(answer_keep)
            }
            for future in as_completed(futures):
                idx = futures[future]
                record = answer_keep[idx]
                result = future.result()
                reasoning_latencies.append(result.latency_ms)
                if result.success:
                    updated = merge_quality_meta(
                        record,
                        {
                            "reasoning_score": int(result.payload.get("reasoning_score", 0)),
                            "reasoning_judge_success": True,
                            "reasoning_judge_error": "",
                            "reasoning_judge_latency_ms": result.latency_ms,
                            "reasoning_judge_payload": result.payload,
                        },
                    )
                else:
                    reasoning_gpt_failures += 1
                    updated = merge_quality_meta(
                        record,
                        {
                            "reasoning_score": 0,
                            "reasoning_judge_success": False,
                            "reasoning_judge_error": result.error,
                            "reasoning_judge_latency_ms": result.latency_ms,
                            "reasoning_judge_payload": {},
                        },
                    )
                stage2_rows[idx] = updated

    reasoning_keep = [
        row for row in stage2_rows if int(row.get("metadata", {}).get("quality_filter", {}).get("reasoning_score", 0)) == 1
    ]
    reasoning_drop = [
        row for row in stage2_rows if int(row.get("metadata", {}).get("quality_filter", {}).get("reasoning_score", 0)) == 0
    ]

    stage2_scored_path = run_dir / f"{split}.stage2_reasoning_scored.jsonl"
    stage2_keep_path = run_dir / f"{split}.stage2_reasoning_keep.jsonl"
    stage2_drop_path = run_dir / f"{split}.stage2_reasoning_drop.jsonl"
    write_jsonl(stage2_scored_path, stage2_rows)
    write_jsonl(stage2_keep_path, reasoning_keep)
    write_jsonl(stage2_drop_path, reasoning_drop)

    # Final materialization into train_data/{source}.
    target_source_dir = train_data_root / source
    final_keep_path = target_source_dir / f"{split}.jsonl"
    dropped_dir = target_source_dir / "dropped"
    final_stage1_drop_path = dropped_dir / f"{split}.stage1_answer_drop.jsonl"
    final_stage2_drop_path = dropped_dir / f"{split}.stage2_reasoning_drop.jsonl"
    write_jsonl(final_keep_path, reasoning_keep)
    write_jsonl(final_stage1_drop_path, answer_drop)
    write_jsonl(final_stage2_drop_path, reasoning_drop)

    summary = {
        "input_path": str(input_path),
        "source": source,
        "split": split,
        "sample_size": len(records),
        "stage1": {
            **summary_stats(stage1_rows, "answer_score"),
            "rule_kept": rule_kept,
            "rule_dropped": rule_dropped,
            "gpt_unresolved_count": len(unresolved_indices),
            "gpt_failure_count": answer_gpt_failures,
            "gpt_average_latency_ms": statistics.mean(answer_latencies) if answer_latencies else 0.0,
            "scored_path": str(stage1_scored_path),
            "keep_path": str(stage1_keep_path),
            "drop_path": str(stage1_drop_path),
        },
        "stage2": {
            **summary_stats(stage2_rows, "reasoning_score"),
            "input_from_stage1_keep": len(answer_keep),
            "gpt_failure_count": reasoning_gpt_failures,
            "gpt_average_latency_ms": statistics.mean(reasoning_latencies) if reasoning_latencies else 0.0,
            "scored_path": str(stage2_scored_path),
            "keep_path": str(stage2_keep_path),
            "drop_path": str(stage2_drop_path),
        },
        "final_outputs": {
            "train_data_keep_path": str(final_keep_path),
            "train_data_stage1_drop_path": str(final_stage1_drop_path),
            "train_data_stage2_drop_path": str(final_stage2_drop_path),
        },
    }

    summary_path = run_dir / f"{split}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
