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


PROMPT_PATH = REPO_ROOT / "prompts" / "financial_phrasebank_answer_judge_prompt.txt"
LABELS = {"negative", "neutral", "positive"}
ABSTAIN_PATTERNS = (
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not enough information",
    "insufficient information",
    "unknown",
)


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-first answer scoring for FinancialPhraseBank.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "financial_phrasebank_full_v2" / "output" / "financial_phrasebank" / "train.jsonl"),
            str(REPO_ROOT / "data" / "financial_phrasebank_full_v2" / "output" / "financial_phrasebank" / "val.jsonl"),
        ],
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports" / "answer_score_full" / "financial_phrasebank"))
    parser.add_argument("--filtered_output_dir", default=str(REPO_ROOT / "data" / "distilled_filtered" / "financial_phrasebank"))
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_concurrency", type=int)
    parser.add_argument("--model_name", default="gpt-5.4")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


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


def truncate_text(text: Any, limit: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def render_prompt(record: dict[str, Any]) -> str:
    template = load_prompt_template()
    return template.format(
        task_domain=truncate_text(record.get("task_domain", ""), 100),
        question=truncate_text(record.get("question", ""), 500),
        context=truncate_text(record.get("context", ""), 2500),
        reference_answer=truncate_text(record.get("reference_answer", ""), 100),
        candidate_answer=truncate_text(record.get("model_answer", ""), 300),
        candidate_reasoning=truncate_text(record.get("model_reasoning", ""), 2500),
    )


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def normalize_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) else 0
    except Exception:  # noqa: BLE001
        return 0


def parse_judge_response(text: str) -> dict[str, Any]:
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
    provider_result = provider.generate(render_prompt(record))
    if not provider_result.success:
        return JudgeResult(False, {}, provider_result.error, provider_result.latency_ms)
    try:
        return JudgeResult(True, parse_judge_response(provider_result.text), "", provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(False, {}, f"parse_error: {exc}", provider_result.latency_ms)


def rule_judge(record: dict[str, Any]) -> dict[str, Any] | None:
    reference = extract_label(record.get("reference_answer", ""))
    candidate_raw = str(record.get("model_answer", "")).strip()
    candidate = extract_label(candidate_raw)

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
            "reason": "The candidate abstains instead of giving a sentiment label.",
            "scoring_mode": "rule",
        }
    if reference and candidate:
        ok = reference == candidate
        return {
            "score": 1 if ok else 0,
            "judgment": "rule_match" if ok else "incorrect",
            "mismatch_type": "none" if ok else "label_error",
            "reason": (
                f"The candidate label {candidate} matches the reference."
                if ok
                else f"The candidate label {candidate} does not match the reference {reference}."
            ),
            "scoring_mode": "rule",
        }
    return None


def shard_suffix(start_index: int, end_index: int) -> str:
    return f".{start_index}_{end_index}"


def judged_output_path(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> Path:
    return output_dir / f"{input_path.stem}{shard_suffix(start_index, end_index)}.answer_scored.jsonl"


def filtered_paths(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> tuple[Path, Path]:
    suffix = shard_suffix(start_index, end_index)
    return output_dir / f"{input_path.stem}{suffix}.keep.jsonl", output_dir / f"{input_path.stem}{suffix}.drop.jsonl"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
