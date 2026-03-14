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


PROMPT_PATH = REPO_ROOT / "prompts" / "lenient_answer_judge_prompt.txt"


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for lenient answer judging."""
    parser = argparse.ArgumentParser(description="Judge lenient answer consistency with the current model.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "distilled" / "convfinqa" / "train.jsonl"),
            str(REPO_ROOT / "data" / "distilled" / "convfinqa" / "val.jsonl"),
        ],
        help="One or more distilled JSONL files to judge.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "reports" / "lenient_answer_judge"))
    parser.add_argument("--max_samples", type=int, help="Limit each input file to the first N rows.")
    parser.add_argument("--max_concurrency", type=int, help="Override API concurrency.")
    parser.add_argument("--include_context", action="store_true", help="Include full context in the judge prompt.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    """Load the judge prompt template once."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def truncate_text(text: str, limit: int = 5000) -> str:
    """Truncate prompt fields to keep judge prompts bounded."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def render_prompt(record: dict[str, Any], *, include_context: bool) -> str:
    """Render the lenient consistency judge prompt for one record."""
    template = load_prompt_template()
    context = record.get("context", "") if include_context else ""
    return template.format(
        question=truncate_text(record.get("question", ""), 2000) or "N/A",
        context=truncate_text(context, 4000) or "N/A",
        reference_answer=truncate_text(record.get("reference_answer", ""), 1000) or "N/A",
        candidate_answer=truncate_text(record.get("model_answer", ""), 1000) or "N/A",
        candidate_reasoning=truncate_text(record.get("model_reasoning", ""), 3000) or "N/A",
    )


def extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from a text blob."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the model judge response into a normalized JSON payload."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    return {
        "is_consistent": bool(payload.get("is_consistent", False)),
        "judgment": str(payload.get("judgment", "")),
        "mismatch_type": str(payload.get("mismatch_type", "")),
        "normalized_reference": str(payload.get("normalized_reference", "")),
        "normalized_candidate": str(payload.get("normalized_candidate", "")),
        "reason": str(payload.get("reason", "")),
    }


def judge_record(
    provider: OpenAICompatibleProvider,
    record: dict[str, Any],
    *,
    include_context: bool,
) -> JudgeResult:
    """Judge one record with the current model."""
    prompt = render_prompt(record, include_context=include_context)
    provider_result = provider.generate(prompt)
    if not provider_result.success:
        return JudgeResult(success=False, payload={}, error=provider_result.error, latency_ms=provider_result.latency_ms)
    try:
        payload = parse_judge_response(provider_result.text)
        return JudgeResult(success=True, payload=payload, error="", latency_ms=provider_result.latency_ms)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(success=False, payload={}, error=f"parse_error: {exc}", latency_ms=provider_result.latency_ms)


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    """Build the output path for one judged file."""
    return output_dir / input_path.parent.name / f"{input_path.stem}.judged.jsonl"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a summary over judged rows."""
    total = len(rows)
    judged_ok = sum(1 for row in rows if row["judge_success"])
    consistent = sum(1 for row in rows if row.get("judge", {}).get("is_consistent"))
    mismatch_counts: dict[str, int] = {}
    judgment_counts: dict[str, int] = {}
    latencies = [row["judge_latency_ms"] for row in rows if row["judge_latency_ms"] > 0]
    for row in rows:
        judge = row.get("judge", {})
        mismatch = judge.get("mismatch_type", "")
        judgment = judge.get("judgment", "")
        if mismatch:
            mismatch_counts[mismatch] = mismatch_counts.get(mismatch, 0) + 1
        if judgment:
            judgment_counts[judgment] = judgment_counts.get(judgment, 0) + 1
    return {
        "total_samples": total,
        "judge_success_count": judged_ok,
        "judge_failure_count": total - judged_ok,
        "lenient_consistent_count": consistent,
        "lenient_consistent_rate": (consistent / judged_ok) if judged_ok else 0.0,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "judgment_counts": judgment_counts,
        "mismatch_type_counts": mismatch_counts,
    }


def main() -> int:
    """Judge lenient answer consistency for one or more distilled files."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    provider = OpenAICompatibleProvider.from_config(config)
    output_dir = Path(args.output_dir).resolve()
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
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(judge_record, provider, record, include_context=args.include_context): (index, record)
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                index, record = futures[future]
                result = future.result()
                rows[index] = {
                    "id": record.get("id", ""),
                    "source": record.get("source", ""),
                    "split": record.get("split", ""),
                    "question": record.get("question", ""),
                    "reference_answer": record.get("reference_answer", ""),
                    "model_answer": record.get("model_answer", ""),
                    "model_reasoning": record.get("model_reasoning", ""),
                    "judge_success": result.success,
                    "judge_error": result.error,
                    "judge_latency_ms": result.latency_ms,
                    "judge": result.payload if result.success else {},
                }

        out_path = output_path_for(input_path, output_dir)
        ensure_parent(out_path)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        file_summary = {
            "input_path": str(input_path),
            "output_path": str(out_path),
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
