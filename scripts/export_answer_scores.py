#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_parent, iter_jsonl


def parse_args() -> argparse.Namespace:
    """Parse CLI args for exporting answer scores from judge output."""
    parser = argparse.ArgumentParser(description="Export binary answer scores from lenient judge output.")
    parser.add_argument("--judged", required=True, help="Path to *.judged.jsonl from run_lenient_answer_judge.py.")
    parser.add_argument("--scored_output", required=True, help="Output path for per-row answer scores.")
    parser.add_argument("--input", help="Original distilled JSONL to split into keep/drop subsets.")
    parser.add_argument("--keep_output", help="Output path for answer_score=1 original rows.")
    parser.add_argument("--drop_output", help="Output path for answer_score=0 original rows.")
    return parser.parse_args()


def answer_score_from_row(row: dict) -> int:
    """Map one judged row to a binary answer score."""
    judge = row.get("judge", {})
    return 1 if row.get("judge_success") and judge.get("is_consistent", False) else 0


def main() -> int:
    """Write binary answer-score rows and optional keep/drop splits."""
    args = parse_args()
    judged_path = Path(args.judged).resolve()
    scored_output = Path(args.scored_output).resolve()

    rows = list(iter_jsonl(judged_path))
    ensure_parent(scored_output)

    score_counts: Counter[int] = Counter()
    judgment_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    judged_by_id: dict[str, int] = {}

    with scored_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            score = answer_score_from_row(row)
            judge = row.get("judge", {})
            payload = {
                "id": row.get("id", ""),
                "source": row.get("source", ""),
                "split": row.get("split", ""),
                "reference_answer": row.get("reference_answer", ""),
                "model_answer": row.get("model_answer", ""),
                "answer_score": score,
                "judge_success": row.get("judge_success", False),
                "judge_error": row.get("judge_error", ""),
                "judge_latency_ms": row.get("judge_latency_ms", 0),
                "judge_judgment": judge.get("judgment", ""),
                "judge_mismatch_type": judge.get("mismatch_type", ""),
                "judge_reason": judge.get("reason", ""),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            sample_id = str(payload["id"])
            if sample_id:
                judged_by_id[sample_id] = score
            score_counts[score] += 1
            judgment_counts[str(payload["judge_judgment"])] += 1
            mismatch_counts[str(payload["judge_mismatch_type"])] += 1

    keep_count = 0
    drop_count = 0
    missing_judge_count = 0

    if args.input or args.keep_output or args.drop_output:
        if not (args.input and args.keep_output and args.drop_output):
            raise SystemExit("--input, --keep_output, and --drop_output must be provided together.")

        input_path = Path(args.input).resolve()
        keep_output = Path(args.keep_output).resolve()
        drop_output = Path(args.drop_output).resolve()
        ensure_parent(keep_output)
        ensure_parent(drop_output)

        with keep_output.open("w", encoding="utf-8") as keep_handle, drop_output.open("w", encoding="utf-8") as drop_handle:
            for record in iter_jsonl(input_path):
                sample_id = str(record.get("id", ""))
                score = judged_by_id.get(sample_id)
                if score is None:
                    missing_judge_count += 1
                    drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    drop_count += 1
                    continue
                if score == 1:
                    keep_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    keep_count += 1
                else:
                    drop_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    drop_count += 1

    summary = {
        "judged_path": str(judged_path),
        "scored_output": str(scored_output),
        "total_rows": len(rows),
        "answer_score_1_count": score_counts.get(1, 0),
        "answer_score_0_count": score_counts.get(0, 0),
        "answer_score_1_rate": (score_counts.get(1, 0) / len(rows)) if rows else 0.0,
        "judgment_counts": dict(judgment_counts),
        "mismatch_type_counts": dict(mismatch_counts),
        "keep_count": keep_count,
        "drop_count": drop_count,
        "missing_judge_count": missing_judge_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
