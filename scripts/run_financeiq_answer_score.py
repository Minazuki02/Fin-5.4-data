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
from src.utils.io import ensure_parent, iter_jsonl


PROMPT_PATH = REPO_ROOT / "prompts" / "financeiq_answer_judge_prompt.txt"
CHOICE_SET = {"A", "B", "C", "D"}
TRUE_SET = {"正确", "对", "是", "true", "yes", "y", "t", "√"}
FALSE_SET = {"错误", "错", "否", "false", "no", "n", "f", "×"}
ABSTAIN_PATTERNS = ("无法确定", "不能确定", "无法判断", "不能判断", "不确定", "不知道", "无法得出", "无法从题干")


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FinanceIQ answer scoring."""
    parser = argparse.ArgumentParser(description="Rule-first answer scoring for FinanceIQ.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "financeiq_full_v3" / "output" / "financeiq" / "train.jsonl"),
            str(REPO_ROOT / "data" / "financeiq_full_v3" / "output" / "financeiq" / "val.jsonl"),
        ],
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "data" / "reports" / "answer_score_full" / "financeiq"),
    )
    parser.add_argument(
        "--filtered_output_dir",
        default=str(REPO_ROOT / "data" / "distilled_filtered" / "financeiq"),
    )
    parser.add_argument("--max_samples", type=int, help="Limit each input file to the first N rows.")
    parser.add_argument("--max_concurrency", type=int, help="Override API concurrency.")
    parser.add_argument("--model_name", default="gpt-5.4", help="Model name override for LLM fallback.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    """Load the LLM fallback judge prompt once."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def truncate_text(text: Any, limit: int) -> str:
    """Truncate prompt fields to keep requests bounded."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def normalize_text(text: Any) -> str:
    """Normalize text for deterministic comparison."""
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    cleaned = cleaned.replace("：", ":").replace("，", ",").replace("。", ".")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def normalize_option_text(text: Any) -> str:
    """Normalize option text after removing leading labels."""
    cleaned = normalize_text(text)
    cleaned = re.sub(r"^[A-D][\.\):：、\s]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def option_map(record: dict[str, Any]) -> dict[str, str]:
    """Build a letter-to-option-text mapping from the record."""
    mapping: dict[str, str] = {}
    options = record.get("options") or []
    if isinstance(options, list):
        for raw in options:
            text = str(raw or "").strip()
            match = re.match(r"^\s*([A-D])[\.\):：、\s]+(.*)$", text, flags=re.IGNORECASE)
            if match:
                mapping[match.group(1).upper()] = match.group(2).strip()
    raw_sample = record.get("metadata", {}).get("raw_sample", {})
    if isinstance(raw_sample, dict):
        for letter in CHOICE_SET:
            if letter in raw_sample and letter not in mapping:
                mapping[letter] = str(raw_sample.get(letter, "")).strip()
    return mapping


def extract_choice_letter(text: Any, mapping: dict[str, str] | None = None) -> str:
    """Extract an A/B/C/D style final answer when possible."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    compact = normalize_text(raw).upper()
    if compact in CHOICE_SET:
        return compact

    patterns = [
        r"^(?:答案|选项|故选|应选|选择|最终答案)[是为:： ]*([A-D])$",
        r"^(?:答案|选项|故选|应选|选择|最终答案)[是为:： ]*([A-D]).*$",
        r"^([A-D])(?:选项)?$",
        r"^选([A-D])$",
        r"^选项([A-D])$",
    ]
    for pattern in patterns:
        match = re.match(pattern, compact)
        if match:
            return match.group(1)

    if mapping:
        normalized = normalize_option_text(raw)
        exact_hits = [letter for letter, option_text in mapping.items() if normalized == normalize_option_text(option_text)]
        if len(exact_hits) == 1:
            return exact_hits[0]
        contains_hits = [
            letter
            for letter, option_text in mapping.items()
            if normalize_option_text(option_text) and normalize_option_text(option_text) in normalized
        ]
        if len(contains_hits) == 1:
            return contains_hits[0]
    return ""


def extract_true_false(text: Any) -> str:
    """Extract a true/false style answer when possible."""
    raw = normalize_text(text)
    if not raw:
        return ""
    if raw in TRUE_SET or any(raw.startswith(token) for token in ("判断为正确", "答案为正确")):
        return "true"
    if raw in FALSE_SET or any(raw.startswith(token) for token in ("判断为错误", "答案为错误")):
        return "false"
    return ""


def is_abstain(text: Any) -> bool:
    """Return whether a candidate answer abstains."""
    raw = str(text or "").strip()
    return any(pattern in raw for pattern in ABSTAIN_PATTERNS)


def render_options_block(record: dict[str, Any]) -> str:
    """Render options for the LLM fallback prompt."""
    mapping = option_map(record)
    if not mapping:
        return "N/A"
    return "\n".join(f"{letter}. {mapping[letter]}" for letter in sorted(mapping))


def render_prompt(record: dict[str, Any]) -> str:
    """Render the fallback answer-judge prompt."""
    template = load_prompt_template()
    return template.format(
        task_domain=truncate_text(record.get("task_domain", ""), 100) or "N/A",
        question=truncate_text(record.get("question", ""), 2500) or "N/A",
        options_block=truncate_text(render_options_block(record), 2500) or "N/A",
        reference_answer=truncate_text(record.get("reference_answer", ""), 300) or "N/A",
        candidate_answer=truncate_text(record.get("model_answer", ""), 500) or "N/A",
        candidate_reasoning=truncate_text(record.get("model_reasoning", ""), 3000) or "N/A",
    )


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from raw text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def normalize_binary(value: Any) -> int:
    """Normalize a score field to 0 or 1."""
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) else 0
    except Exception:  # noqa: BLE001
        return 0


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse normalized JSON from the fallback judge."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    return {
        "score": normalize_binary(payload.get("score", 0)),
        "judgment": str(payload.get("judgment", "")),
        "mismatch_type": str(payload.get("mismatch_type", "")),
        "reason": str(payload.get("reason", "")),
        "scoring_mode": "llm",
    }


def llm_judge(provider: OpenAICompatibleProvider, record: dict[str, Any]) -> JudgeResult:
    """Judge one record with the fallback LLM."""
    prompt = render_prompt(record)
    provider_result = provider.generate(prompt)
    if not provider_result.success:
        return JudgeResult(success=False, payload={}, error=provider_result.error, latency_ms=provider_result.latency_ms)
    try:
        payload = parse_judge_response(provider_result.text)
        return JudgeResult(success=True, payload=payload, error="", latency_ms=provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(success=False, payload={}, error=f"parse_error: {exc}", latency_ms=provider_result.latency_ms)


def rule_judge(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a deterministic judgment when the sample can be matched by rules."""
    reference_raw = str(record.get("reference_answer", "")).strip()
    candidate_raw = str(record.get("model_answer", "")).strip()
    mapping = option_map(record)

    if not reference_raw:
        return None
    if not candidate_raw:
        return {
            "score": 0,
            "judgment": "incorrect",
            "mismatch_type": "format_error",
            "reason": "The candidate answer is empty.",
            "scoring_mode": "rule",
        }
    if is_abstain(candidate_raw):
        return {
            "score": 0,
            "judgment": "incorrect",
            "mismatch_type": "abstain_error",
            "reason": "The candidate abstains instead of giving a final answer.",
            "scoring_mode": "rule",
        }

    reference_choice = extract_choice_letter(reference_raw, mapping)
    candidate_choice = extract_choice_letter(candidate_raw, mapping)
    if reference_choice:
        if not candidate_choice:
            return None
        ok = reference_choice == candidate_choice
        return {
            "score": 1 if ok else 0,
            "judgment": "rule_match" if ok else "incorrect",
            "mismatch_type": "none" if ok else "option_error",
            "reason": (
                f"The candidate selects option {candidate_choice}, which matches the reference."
                if ok
                else f"The candidate selects option {candidate_choice}, but the reference is {reference_choice}."
            ),
            "scoring_mode": "rule",
        }

    reference_tf = extract_true_false(reference_raw)
    candidate_tf = extract_true_false(candidate_raw)
    if reference_tf:
        if not candidate_tf:
            return None
        ok = reference_tf == candidate_tf
        return {
            "score": 1 if ok else 0,
            "judgment": "rule_match" if ok else "incorrect",
            "mismatch_type": "none" if ok else "judgment_error",
            "reason": "The candidate gives the same judgment as the reference." if ok else "The candidate gives the opposite judgment.",
            "scoring_mode": "rule",
        }

    return None


def shard_suffix(start_index: int, end_index: int) -> str:
    """Return a stable filename suffix for a shard."""
    return f".{start_index}_{end_index}"


def judged_output_path(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> Path:
    """Return the judged JSONL path for one source file."""
    return output_dir / f"{input_path.stem}{shard_suffix(start_index, end_index)}.answer_scored.jsonl"


def filtered_paths(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> tuple[Path, Path]:
    """Return keep/drop output paths for answer filtering."""
    suffix = shard_suffix(start_index, end_index)
    return output_dir / f"{input_path.stem}{suffix}.keep.jsonl", output_dir / f"{input_path.stem}{suffix}.drop.jsonl"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize answer-scoring results."""
    total = len(rows)
    success = sum(1 for row in rows if row["judge_success"])
    keep = sum(1 for row in rows if row.get("judge", {}).get("score") == 1)
    rule_count = sum(1 for row in rows if row.get("judge", {}).get("scoring_mode") == "rule")
    llm_count = sum(1 for row in rows if row.get("judge", {}).get("scoring_mode") == "llm")
    latencies = [row["judge_latency_ms"] for row in rows if row["judge_latency_ms"] > 0]
    mismatch_counts: dict[str, int] = {}
    for row in rows:
        mismatch = row.get("judge", {}).get("mismatch_type", "")
        if mismatch:
            mismatch_counts[mismatch] = mismatch_counts.get(mismatch, 0) + 1
    return {
        "total_samples": total,
        "judge_success_count": success,
        "judge_failure_count": total - success,
        "keep_count": keep,
        "keep_rate": (keep / success) if success else 0.0,
        "rule_scored_count": rule_count,
        "llm_scored_count": llm_count,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "mismatch_type_counts": mismatch_counts,
    }


def main() -> int:
    """Run FinanceIQ answer scoring and split keep/drop outputs."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    provider = OpenAICompatibleProvider.from_config(config)
    provider.model_name = args.model_name

    output_dir = Path(args.output_dir).resolve()
    filtered_dir = Path(args.filtered_output_dir).resolve()
    max_concurrency = args.max_concurrency or int(config["api"]["max_concurrency"])

    all_summary: dict[str, Any] = {
        "files": [],
        "model_name": provider.model_name,
        "api_base": provider.api_base,
        "prompt_path": str(PROMPT_PATH),
        "max_concurrency": max_concurrency,
    }

    for raw_input in args.inputs:
        input_path = Path(raw_input).resolve()
        records = list(iter_jsonl(input_path))
        if args.max_samples is not None:
            records = records[: args.max_samples]

        rows: list[dict[str, Any]] = [None] * len(records)  # type: ignore[list-item]
        llm_jobs: list[tuple[int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            rule_result = rule_judge(record)
            if rule_result is not None:
                rows[index] = {
                    "id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "split": record.get("split", ""),
                    "task_domain": record.get("task_domain", ""),
                    "question": record.get("question", ""),
                    "reference_answer": record.get("reference_answer", ""),
                    "model_answer": record.get("model_answer", ""),
                    "model_reasoning": record.get("model_reasoning", ""),
                    "judge_success": True,
                    "judge_error": "",
                    "judge_latency_ms": 0,
                    "judge": rule_result,
                }
            else:
                llm_jobs.append((index, record))

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(llm_judge, provider, record): (index, record) for index, record in llm_jobs}
            completed = 0
            for future in as_completed(futures):
                index, record = futures[future]
                result = future.result()
                rows[index] = {
                    "id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "split": record.get("split", ""),
                    "task_domain": record.get("task_domain", ""),
                    "question": record.get("question", ""),
                    "reference_answer": record.get("reference_answer", ""),
                    "model_answer": record.get("model_answer", ""),
                    "model_reasoning": record.get("model_reasoning", ""),
                    "judge_success": result.success,
                    "judge_error": result.error,
                    "judge_latency_ms": result.latency_ms,
                    "judge": result.payload if result.success else {},
                }
                completed += 1
                if completed % 256 == 0 or completed == len(llm_jobs):
                    print(
                        json.dumps(
                            {
                                "input_path": str(input_path),
                                "phase": "answer_score",
                                "completed_llm_jobs": completed,
                                "total_llm_jobs": len(llm_jobs),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

        judged_path = judged_output_path(input_path, output_dir, 0, len(records))
        ensure_parent(judged_path)
        with judged_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        keep_path, drop_path = filtered_paths(input_path, filtered_dir, 0, len(records))
        ensure_parent(keep_path)
        ensure_parent(drop_path)
        keep_count = 0
        drop_count = 0
        with keep_path.open("w", encoding="utf-8") as keep_handle, drop_path.open("w", encoding="utf-8") as drop_handle:
            for record, row in zip(records, rows):
                if row["judge_success"] and row.get("judge", {}).get("score") == 1:
                    keep_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    keep_count += 1
                else:
                    drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    drop_count += 1

        file_summary = {
            "input_path": str(input_path),
            "judged_output_path": str(judged_path),
            "keep_output": str(keep_path),
            "drop_output": str(drop_path),
            "keep_count": keep_count,
            "drop_count": drop_count,
            **summarize(rows),
        }
        all_summary["files"].append(file_summary)

    summary_path = output_dir / "summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
