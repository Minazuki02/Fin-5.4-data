#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.distill.parser import parse_response
from src.distill.prompt_builder import build_prompt
from src.distill.provider import OpenAICompatibleProvider
from src.eval.convfinqa_matcher import match_convfinqa_record
from src.postprocess.convfinqa_postprocess import postprocess_convfinqa_answer, question_is_numeric
from src.utils.io import ensure_parent, iter_jsonl
from src.utils.logging_utils import configure_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ConvFinQA-only repair."""
    parser = argparse.ArgumentParser(description="Repair ConvFinQA distilled answers selectively.")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "distilled" / "convfinqa" / "train.jsonl"))
    parser.add_argument("--output", default="")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--prompt_version", default="convfinqa_repair_v1")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_repairs", type=int)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe_logger() -> logging.Logger:
    """Prefer the shared file logger but fall back to stderr when writes are blocked."""
    log_path = REPO_ROOT / "data" / "logs" / "convfinqa_repair.log"
    try:
        return configure_logger(log_path)
    except PermissionError:
        logger = logging.getLogger("convfinqa_repair")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)
        return logger


def inconsistency_needs_repair(record: dict[str, Any]) -> bool:
    """Detect when reasoning and final answer diverge enough to justify repair."""
    post = postprocess_convfinqa_answer(
        question=str(record.get("question", "")),
        model_answer=str(record.get("model_answer", "")),
        reasoning=str(record.get("model_reasoning", "")),
    )
    if not post.normalized_reasoning_answer or not post.normalized_model_answer:
        return False
    return not post.reasoning_answer_consistent


def repair_flags(record: dict[str, Any]) -> list[str]:
    """Return the repair reasons for one ConvFinQA record."""
    question = str(record.get("question", ""))
    answer = str(record.get("model_answer", ""))
    post = postprocess_convfinqa_answer(question=question, model_answer=answer, reasoning=str(record.get("model_reasoning", "")))
    match = match_convfinqa_record(record)

    reasons: list[str] = []
    if not answer.strip():
        reasons.append("empty_answer")
    if match.mismatch_type == "rounding_or_format":
        reasons.append("strict_fail_lenient_pass")
    if post.is_abstain:
        reasons.append("abstain_answer")
    if question_is_numeric(question) and not post.numeric_parseable:
        reasons.append("numeric_not_normalizable")
    if inconsistency_needs_repair(record):
        reasons.append("reasoning_answer_inconsistent")
    return reasons


def apply_local_normalization(record: dict[str, Any], reasons: list[str]) -> dict[str, Any] | None:
    """Repair locally when normalization alone is enough."""
    if "strict_fail_lenient_pass" not in reasons:
        return None

    updated = dict(record)
    post = postprocess_convfinqa_answer(
        question=str(record.get("question", "")),
        model_answer=str(record.get("model_answer", "")),
        reasoning=str(record.get("model_reasoning", "")),
    )
    if not post.normalized_model_answer:
        return None
    updated["model_answer"] = post.normalized_model_answer
    metadata = dict(updated.get("metadata", {}))
    repair_meta = dict(metadata.get("convfinqa_repair", {}))
    repair_meta.update(
        {
            "repair_mode": "local_normalize",
            "repair_reasons": reasons,
            "original_model_answer": str(record.get("model_answer", "")),
            "normalized_model_answer": post.normalized_model_answer,
        }
    )
    metadata["convfinqa_repair"] = repair_meta
    updated["metadata"] = metadata
    return updated


def repair_with_model(
    record: dict[str, Any],
    *,
    provider: OpenAICompatibleProvider,
    prompt_version: str,
) -> dict[str, Any]:
    """Re-query the model with the ConvFinQA-specific prompt and normalize the result."""
    prompt = build_prompt(record, prompt_version=prompt_version)
    provider_result = provider.generate(prompt)

    updated = dict(record)
    metadata = dict(updated.get("metadata", {}))
    repair_meta = dict(metadata.get("convfinqa_repair", {}))
    repair_meta.update(
        {
            "repair_mode": "model_regenerate",
            "teacher_model": provider.model_name,
            "api_success": provider_result.success,
            "api_latency_ms": provider_result.latency_ms,
            "raw_response_text": provider_result.text,
            "parse_error": "",
            "normalized_model_answer": "",
        }
    )

    if provider_result.success:
        parsed = parse_response(provider_result.text, updated)
        repair_meta["parse_status"] = parsed.status
        repair_meta["parse_error"] = parsed.error
        updated["model_reasoning"] = parsed.reasoning if parsed.reasoning else str(record.get("model_reasoning", ""))
        post = postprocess_convfinqa_answer(
            question=str(updated.get("question", "")),
            model_answer=parsed.final_answer,
            reasoning=updated["model_reasoning"],
        )
        updated["model_answer"] = post.normalized_model_answer or parsed.final_answer.strip()
        repair_meta["normalized_model_answer"] = post.normalized_model_answer
    else:
        repair_meta["parse_status"] = "api_failed"
        repair_meta["parse_error"] = provider_result.error

    metadata["convfinqa_repair"] = repair_meta
    updated["metadata"] = metadata
    return updated


def repair_task(
    item: tuple[int, dict[str, Any], list[str]],
    *,
    provider: OpenAICompatibleProvider,
    prompt_version: str,
) -> tuple[int, dict[str, Any], list[str]]:
    """Repair one queued model candidate and preserve its target index."""
    index, record, reasons = item
    repaired = repair_with_model(record, provider=provider, prompt_version=prompt_version)
    repaired_metadata = dict(repaired.get("metadata", {}))
    repaired_metadata.setdefault("convfinqa_repair", {})["repair_reasons"] = reasons
    repaired["metadata"] = repaired_metadata
    return index, repaired, reasons


def main() -> int:
    """Run selective ConvFinQA repair."""
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else input_path

    config = load_config(Path(args.config).resolve())
    logger = safe_logger()
    provider = None if args.dry_run else OpenAICompatibleProvider.from_config(config)
    max_concurrency = int(config.get("api", {}).get("max_concurrency", 8))

    records = list(iter_jsonl(input_path))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    to_write: list[dict[str, Any]] = []
    repair_count = 0
    local_fix_count = 0
    model_fix_count = 0
    skipped_count = 0
    candidate_count = 0
    model_candidates: list[tuple[int, dict[str, Any], list[str]]] = []

    for index, record in enumerate(records, 1):
        if str(record.get("source", "")).lower() != "convfinqa":
            raise ValueError("run_convfinqa_repair.py only supports ConvFinQA records")

        reasons = repair_flags(record)
        if not reasons:
            to_write.append(record)
            skipped_count += 1
            continue

        candidate_count += 1
        updated = apply_local_normalization(record, reasons)
        if updated is not None:
            to_write.append(updated)
            repair_count += 1
            local_fix_count += 1
            continue

        if args.dry_run:
            metadata = dict(record.get("metadata", {}))
            metadata["convfinqa_repair"] = {
                "repair_mode": "dry_run_candidate",
                "repair_reasons": reasons,
            }
            dry_record = dict(record)
            dry_record["metadata"] = metadata
            to_write.append(dry_record)
            continue

        if args.max_repairs is not None and len(model_candidates) >= args.max_repairs and not args.overwrite:
            to_write.append(record)
            continue

        model_candidates.append((len(to_write), record, reasons))
        to_write.append(record)

        if args.progress_every > 0 and index % max(args.progress_every * 5, 500) == 0:
            logger.info(
                "convfinqa scan progress: processed=%s/%s candidates=%s local_fix=%s queued_model_fix=%s skipped=%s",
                index,
                len(records),
                candidate_count,
                local_fix_count,
                len(model_candidates),
                skipped_count,
            )

    if model_candidates and not args.dry_run:
        logger.info(
            "convfinqa model repair start: queued=%s concurrency=%s",
            len(model_candidates),
            max_concurrency,
        )
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_map = {
                executor.submit(
                    repair_task,
                    item,
                    provider=provider,
                    prompt_version=args.prompt_version,
                ): item
                for item in model_candidates
            }
            for future in as_completed(future_map):
                target_index, repaired, _ = future.result()
                to_write[target_index] = repaired
                repair_count += 1
                model_fix_count += 1
                if args.progress_every > 0 and model_fix_count % args.progress_every == 0:
                    logger.info(
                        "convfinqa repair progress: model_fix=%s/%s local_fix=%s total_records=%s",
                        model_fix_count,
                        len(model_candidates),
                        local_fix_count,
                        len(records),
                    )

    if output_path == input_path:
        tmp_path = output_path.with_suffix(output_path.suffix + ".repair.tmp")
    else:
        tmp_path = output_path
    ensure_parent(tmp_path)
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in to_write:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if output_path == input_path:
        tmp_path.replace(output_path)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_records": len(records),
        "candidate_count": candidate_count,
        "repair_count": repair_count,
        "local_fix_count": local_fix_count,
        "model_fix_count": model_fix_count,
        "skipped_count": skipped_count,
        "dry_run": args.dry_run,
    }
    logger.info("convfinqa repair finished: %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
