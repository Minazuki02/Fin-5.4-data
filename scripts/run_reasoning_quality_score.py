#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


PROMPT_PATH = REPO_ROOT / "prompts" / "reasoning_quality_score_prompt.txt"


@dataclass(frozen=True)
class ScoreResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    """Parse CLI args for second-round reasoning-quality scoring."""
    parser = argparse.ArgumentParser(description="Run second-round reasoning quality scoring with gpt-5.4.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "distilled_filtered" / "convfinqa" / "train.keep.jsonl"),
            str(REPO_ROOT / "data" / "distilled_filtered" / "convfinqa" / "val.keep.jsonl"),
        ],
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports" / "reasoning_quality_score"))
    parser.add_argument("--filtered_output_dir", default=str(REPO_ROOT / "data" / "distilled_filtered_round2" / "convfinqa"))
    parser.add_argument("--start_index", type=int, default=0, help="0-based inclusive start index for each input file.")
    parser.add_argument("--end_index", type=int, help="0-based exclusive end index for each input file.")
    parser.add_argument("--max_samples", type=int, help="Limit each input file to the first N rows.")
    parser.add_argument("--max_concurrency", type=int, help="Override API concurrency.")
    parser.add_argument("--model_name", default="gpt-5.4", help="Model name override for scoring.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    """Load the second-round scoring prompt template."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def truncate_text(text: Any, limit: int) -> str:
    """Truncate long prompt fields to bound cost."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def render_prompt(record: dict[str, Any]) -> str:
    """Render the reasoning-quality scoring prompt."""
    template = load_prompt_template()
    return template.format(
        task_domain=truncate_text(record.get("task_domain", ""), 200),
        question=truncate_text(record.get("question", ""), 2000),
        reference_answer=truncate_text(record.get("reference_answer", ""), 500),
        candidate_answer=truncate_text(record.get("model_answer", ""), 500),
        candidate_reasoning=truncate_text(record.get("model_reasoning", ""), 4000),
    )


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from a response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def normalize_binary(value: Any) -> int:
    """Normalize one score field to 0 or 1."""
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) else 0
    except Exception:  # noqa: BLE001
        return 0


def parse_score_response(text: str) -> dict[str, Any]:
    """Parse scoring JSON from the model."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    return {
        "score": normalize_binary(payload.get("score", 0)),
        "internal_consistency": normalize_binary(payload.get("internal_consistency", 0)),
        "term_overlap": normalize_binary(payload.get("term_overlap", 0)),
        "step_count_ok": normalize_binary(payload.get("step_count_ok", 0)),
        "logical_consistency": normalize_binary(payload.get("logical_consistency", 0)),
        "content_diversity": normalize_binary(payload.get("content_diversity", 0)),
        "domain_relevance": normalize_binary(payload.get("domain_relevance", 0)),
        "instruction_alignment": normalize_binary(payload.get("instruction_alignment", 0)),
        "reason": str(payload.get("reason", "")),
    }


def score_record(provider: OpenAICompatibleProvider, record: dict[str, Any]) -> ScoreResult:
    """Score one record with the current model."""
    prompt = render_prompt(record)
    provider_result = provider.generate(prompt)
    if not provider_result.success:
        return ScoreResult(success=False, payload={}, error=provider_result.error, latency_ms=provider_result.latency_ms)
    try:
        payload = parse_score_response(provider_result.text)
        return ScoreResult(success=True, payload=payload, error="", latency_ms=provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return ScoreResult(success=False, payload={}, error=f"parse_error: {exc}", latency_ms=provider_result.latency_ms)


def shard_suffix(start_index: int, end_index: int) -> str:
    """Return a stable filename suffix for a shard."""
    return f".{start_index}_{end_index}"


def judged_output_path(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> Path:
    """Return the judged JSONL path for one source file."""
    return output_dir / input_path.parent.name / f"{input_path.stem}{shard_suffix(start_index, end_index)}.scored.jsonl"


def filtered_paths(input_path: Path, output_dir: Path, start_index: int, end_index: int) -> tuple[Path, Path]:
    """Return keep/drop output paths for round-two filtering."""
    suffix = shard_suffix(start_index, end_index)
    return output_dir / f"{input_path.stem}{suffix}.round2.keep.jsonl", output_dir / f"{input_path.stem}{suffix}.round2.drop.jsonl"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize second-round scoring results."""
    total = len(rows)
    success_count = sum(1 for row in rows if row["score_success"])
    keep_count = sum(1 for row in rows if row.get("score", {}).get("score") == 1)
    latencies = [row["score_latency_ms"] for row in rows if row["score_latency_ms"] > 0]
    reason_counts: dict[str, int] = {}
    for row in rows:
        if not row["score_success"]:
            reason_counts["score_failure"] = reason_counts.get("score_failure", 0) + 1
            continue
        score_value = row.get("score", {}).get("score", 0)
        key = "high_quality" if score_value == 1 else "low_quality"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "total_samples": total,
        "score_success_count": success_count,
        "score_failure_count": total - success_count,
        "high_quality_count": keep_count,
        "high_quality_rate": (keep_count / success_count) if success_count else 0.0,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "outcome_counts": reason_counts,
    }


def main() -> int:
    """Run second-round reasoning-quality scoring and split keep/drop outputs."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    provider = OpenAICompatibleProvider.from_config(config)
    provider.model_name = args.model_name

    report_dir = Path(args.output_dir).resolve()
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
        start_index = max(args.start_index, 0)
        end_index = args.end_index if args.end_index is not None else len(records)
        records = records[start_index:end_index]
        if args.max_samples is not None:
            records = records[: args.max_samples]

        rows: list[dict[str, Any]] = [None] * len(records)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(score_record, provider, record): (index, record) for index, record in enumerate(records)}
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
                    "score_success": result.success,
                    "score_error": result.error,
                    "score_latency_ms": result.latency_ms,
                    "score": result.payload if result.success else {},
                }
                completed += 1
                if completed % 256 == 0 or completed == len(records):
                    print(
                        json.dumps(
                            {
                                "input_path": str(input_path),
                                "phase": "reasoning_score",
                                "completed_jobs": completed,
                                "total_jobs": len(records),
                                "start_index": start_index,
                                "end_index": end_index,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

        judged_path = judged_output_path(input_path, report_dir, start_index, end_index)
        ensure_parent(judged_path)
        with judged_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        keep_path, drop_path = filtered_paths(input_path, filtered_dir, start_index, end_index)
        ensure_parent(keep_path)
        ensure_parent(drop_path)
        keep_count = 0
        drop_count = 0
        with keep_path.open("w", encoding="utf-8") as keep_handle, drop_path.open("w", encoding="utf-8") as drop_handle:
            for record, row in zip(records, rows):
                if row["score_success"] and row.get("score", {}).get("score") == 1:
                    keep_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    keep_count += 1
                else:
                    drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    drop_count += 1

        file_summary = {
            "input_path": str(input_path),
            "judged_output_path": str(judged_path),
            "round2_keep_output": str(keep_path),
            "round2_drop_output": str(drop_path),
            "start_index": start_index,
            "end_index": end_index,
            "round2_keep_count": keep_count,
            "round2_drop_count": drop_count,
            **summarize(rows),
        }
        all_summary["files"].append(file_summary)

    summary_path = report_dir / "summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
