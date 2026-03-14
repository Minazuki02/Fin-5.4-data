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


PROMPT_PATH = REPO_ROOT / "prompts" / "fincuge_instruction_answer_judge_prompt.txt"
CATEGORY_LABELS = [
    "公司",
    "行业",
    "大盘",
    "中国",
    "国际",
    "经济",
    "政策",
    "期货",
    "债券",
    "房地产",
    "外汇",
    "虚拟货币",
    "新冠",
    "能源",
    "政治",
]
SENTIMENT_MAP = {
    "积极": "积极",
    "正面": "积极",
    "正向": "积极",
    "利好": "积极",
    "消极": "消极",
    "负面": "消极",
    "负向": "消极",
    "利空": "消极",
    "中性": "中性",
}
ABSTAIN_PATTERNS = [
    "不能发现具体信息",
    "无法确定",
    "无法判断",
    "未提及",
    "没有提及",
    "没有具体信息",
    "无法从文中",
    "无法从材料",
]


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    payload: dict[str, Any]
    error: str
    latency_ms: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FinCUGE-Instruction answer scoring."""
    parser = argparse.ArgumentParser(description="Rule-first answer scoring for FinCUGE-Instruction.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            str(REPO_ROOT / "data" / "distilled" / "fincuge_instruction" / "train.jsonl"),
            str(REPO_ROOT / "data" / "distilled" / "fincuge_instruction" / "val.jsonl"),
        ],
        help="One or more distilled JSONL files to score.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "data" / "reports" / "answer_score_full" / "fincuge_instruction"),
    )
    parser.add_argument(
        "--filtered_output_dir",
        default=str(REPO_ROOT / "data" / "distilled_filtered" / "fincuge_instruction"),
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


def truncate_text(text: str, limit: int = 4000) -> str:
    """Truncate prompt fields to keep judge prompts bounded."""
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def raw_task_name(record: dict[str, Any]) -> str:
    """Return the upstream task id for one record."""
    return str(record.get("metadata", {}).get("raw_sample", {}).get("task", "")).strip()


def raw_task_desc(record: dict[str, Any]) -> str:
    """Return the upstream task description for one record."""
    return str(record.get("metadata", {}).get("raw_sample", {}).get("desc", "")).strip()


def render_prompt(record: dict[str, Any]) -> str:
    """Render the fallback answer-judge prompt."""
    template = load_prompt_template()
    return template.format(
        raw_task_name=truncate_text(raw_task_name(record), 100) or "N/A",
        raw_task_desc=truncate_text(raw_task_desc(record), 200) or "N/A",
        task_domain=truncate_text(record.get("task_domain", ""), 100) or "N/A",
        question=truncate_text(record.get("question", ""), 2000) or "N/A",
        context=truncate_text(record.get("context", ""), 4000) or "N/A",
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
    """Parse normalized JSON from the fallback judge."""
    blob = extract_json_object(text.strip()) or text.strip()
    payload = json.loads(blob)
    score = int(payload.get("score", 0))
    return {
        "answer_score": 1 if score else 0,
        "judgment": str(payload.get("judgment", "")),
        "mismatch_type": str(payload.get("mismatch_type", "")),
        "reason": str(payload.get("reason", "")),
        "judge_method": "llm",
    }


def normalize_text(text: str) -> str:
    """Normalize general comparison text."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("“", "").replace("”", "").replace('"', "")
    cleaned = cleaned.strip("。；;，,：:、!！?？")
    return cleaned


def normalize_short_answer(text: str) -> str:
    """Normalize wrappers that often appear in FinCUGE short answers."""
    cleaned = normalize_text(text)
    patterns = [
        r"^(以上文本情感极性属于|所属情感是|情感是|情感倾向是|情感倾向为|文本情感是)",
        r"^(该新闻属于|以上新闻类别是|我们把以上新闻归属于|新闻类别是|属于)",
        r"^(和.+?相关的主体是|相关主体是|主体是|答案是)",
        r"^(关系是|联系是)",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip("。；;，,：:、!！?？")


def normalize_sentiment(text: str) -> str:
    """Normalize sentiment labels to the dataset inventory."""
    cleaned = normalize_short_answer(text)
    for key, value in SENTIMENT_MAP.items():
        if key in cleaned:
            return value
    return cleaned


def normalize_category(text: str) -> str:
    """Normalize category answers to the closed label set."""
    cleaned = normalize_short_answer(text)
    for label in CATEGORY_LABELS:
        if label in cleaned:
            return label
    return cleaned


def is_abstain(text: str) -> bool:
    """Return whether the text expresses that no answer is available."""
    cleaned = normalize_short_answer(text)
    return any(pattern in cleaned for pattern in ABSTAIN_PATTERNS)


def is_summary_or_title_task(record: dict[str, Any]) -> bool:
    """Return whether this sample is a title/summary generation task."""
    task = raw_task_name(record)
    question = str(record.get("question", "")).strip()
    return task == "FINNA" or any(token in question for token in ("生成标题", "生成摘要", "摘要", "简述", "总结"))


def looks_rule_eligible(record: dict[str, Any]) -> bool:
    """Return whether the record is eligible for deterministic rule matching."""
    if is_summary_or_title_task(record):
        return False
    reference = str(record.get("reference_answer", "")).strip()
    return len(reference) <= 64


def exact_or_containment_match(reference: str, candidate: str) -> bool:
    """Allow exact match or light span-extension match for short extraction tasks."""
    if not reference or not candidate:
        return False
    if reference == candidate:
        return True
    return reference in candidate or candidate in reference


def rule_judge(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a deterministic judgment when the task can be matched by rules."""
    if not looks_rule_eligible(record):
        return None

    task = raw_task_name(record)
    reference_raw = str(record.get("reference_answer", "")).strip()
    candidate_raw = str(record.get("model_answer", "")).strip()
    if not reference_raw or not candidate_raw:
        return {
            "answer_score": 0,
            "judgment": "rule_mismatch",
            "mismatch_type": "incompleteness",
            "reason": "The candidate answer is empty or missing.",
            "judge_method": "rule",
        }

    if task == "FINFE":
        reference = normalize_sentiment(reference_raw)
        candidate = normalize_sentiment(candidate_raw)
        return {
            "answer_score": 1 if reference == candidate else 0,
            "judgment": "rule_match" if reference == candidate else "rule_mismatch",
            "mismatch_type": "none" if reference == candidate else "sentiment_error",
            "reason": "Sentiment label matched by normalization." if reference == candidate else "Sentiment label differs after normalization.",
            "judge_method": "rule",
        }

    if task == "FINNL":
        reference = normalize_category(reference_raw)
        candidate = normalize_category(candidate_raw)
        return {
            "answer_score": 1 if reference == candidate else 0,
            "judgment": "rule_match" if reference == candidate else "rule_mismatch",
            "mismatch_type": "none" if reference == candidate else "category_error",
            "reason": "Category label matched by normalization." if reference == candidate else "Category label differs after normalization.",
            "judge_method": "rule",
        }

    reference = normalize_short_answer(reference_raw)
    candidate = normalize_short_answer(candidate_raw)
    if is_abstain(reference_raw) or is_abstain(candidate_raw):
        matched = is_abstain(reference_raw) and is_abstain(candidate_raw)
        return {
            "answer_score": 1 if matched else 0,
            "judgment": "rule_match" if matched else "rule_mismatch",
            "mismatch_type": "none" if matched else "abstain_error",
            "reason": "Both answers indicate missing information." if matched else "Only one side indicates missing information.",
            "judge_method": "rule",
        }

    if exact_or_containment_match(reference, candidate):
        return {
            "answer_score": 1,
            "judgment": "rule_match",
            "mismatch_type": "none",
            "reason": "Short-answer span matched by exact or containment rule.",
            "judge_method": "rule",
        }

    mismatch_type = "label_error"
    if task == "FINRE":
        mismatch_type = "relation_error"
    elif task == "FINCQA":
        mismatch_type = "event_error"
    elif task in {"FINQA", "FINESE"}:
        mismatch_type = "entity_error"

    return {
        "answer_score": 0,
        "judgment": "rule_mismatch",
        "mismatch_type": mismatch_type,
        "reason": "Short-answer task failed deterministic normalization and span match.",
        "judge_method": "rule",
    }


def llm_judge_record(provider: OpenAICompatibleProvider, record: dict[str, Any]) -> JudgeResult:
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


def scored_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build the per-input scored JSONL path."""
    return output_dir / f"{input_path.stem}.answer_scored.jsonl"


def filtered_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build keep/drop paths for one input file."""
    return output_dir / f"{input_path.stem}.keep.jsonl", output_dir / f"{input_path.stem}.drop.jsonl"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary statistics over scored rows."""
    total = len(rows)
    score_1 = sum(1 for row in rows if row.get("answer_score") == 1)
    score_success = sum(1 for row in rows if row.get("judge_success"))
    methods: dict[str, int] = {}
    judgments: dict[str, int] = {}
    mismatches: dict[str, int] = {}
    latencies = [row["judge_latency_ms"] for row in rows if row["judge_latency_ms"] > 0]
    for row in rows:
        method = str(row.get("judge_method", ""))
        judgment = str(row.get("judge_judgment", ""))
        mismatch = str(row.get("judge_mismatch_type", ""))
        if method:
            methods[method] = methods.get(method, 0) + 1
        if judgment:
            judgments[judgment] = judgments.get(judgment, 0) + 1
        if mismatch:
            mismatches[mismatch] = mismatches.get(mismatch, 0) + 1
    return {
        "total_rows": total,
        "answer_score_1_count": score_1,
        "answer_score_0_count": total - score_1,
        "answer_score_1_rate": (score_1 / total) if total else 0.0,
        "judge_success_count": score_success,
        "judge_failure_count": total - score_success,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "judge_method_counts": methods,
        "judgment_counts": judgments,
        "mismatch_type_counts": mismatches,
    }


def main() -> int:
    """Run rule-first answer scoring and keep/drop export for FinCUGE-Instruction."""
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
        if args.max_samples is not None:
            records = records[: args.max_samples]

        rows: list[dict[str, Any]] = [None] * len(records)  # type: ignore[list-item]
        fallback_jobs: list[tuple[int, dict[str, Any]]] = []

        for index, record in enumerate(records):
            ruled = rule_judge(record)
            if ruled is None:
                fallback_jobs.append((index, record))
                continue
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
                "judge_method": ruled["judge_method"],
                "judge_judgment": ruled["judgment"],
                "judge_mismatch_type": ruled["mismatch_type"],
                "judge_reason": ruled["reason"],
                "answer_score": ruled["answer_score"],
            }

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(llm_judge_record, provider, record): (index, record) for index, record in fallback_jobs}
            for future in as_completed(futures):
                index, record = futures[future]
                result = future.result()
                payload = result.payload if result.success else {}
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
                    "judge_method": payload.get("judge_method", "llm"),
                    "judge_judgment": payload.get("judgment", ""),
                    "judge_mismatch_type": payload.get("mismatch_type", ""),
                    "judge_reason": payload.get("reason", ""),
                    "answer_score": payload.get("answer_score", 0) if result.success else 0,
                }

        scored_path = scored_output_path(input_path, report_dir)
        keep_path, drop_path = filtered_paths(input_path, filtered_dir)
        ensure_parent(scored_path)
        ensure_parent(keep_path)
        ensure_parent(drop_path)

        keep_count = 0
        drop_count = 0
        with (
            scored_path.open("w", encoding="utf-8") as score_handle,
            keep_path.open("w", encoding="utf-8") as keep_handle,
            drop_path.open("w", encoding="utf-8") as drop_handle,
        ):
            for record, row in zip(records, rows):
                score_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row.get("answer_score") == 1:
                    keep_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    keep_count += 1
                else:
                    drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    drop_count += 1

        file_summary = {
            "input_path": str(input_path),
            "scored_output": str(scored_path),
            "keep_output": str(keep_path),
            "drop_output": str(drop_path),
            "keep_count": keep_count,
            "drop_count": drop_count,
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
