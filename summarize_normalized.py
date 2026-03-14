#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from adapters.common import NORMALIZED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize normalized jsonl files.")
    parser.add_argument(
        "--input-dir",
        default=str(NORMALIZED_ROOT),
        help="Directory containing normalized jsonl files.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to save the summary JSON.",
    )
    return parser.parse_args()


def summarize_file(jsonl_path: Path) -> dict[str, Any]:
    task_type_counter: Counter[str] = Counter()
    task_domain_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    total = 0
    empty_context = 0
    empty_reasoning = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            total += 1
            task_type_counter.update([sample.get("task_type", "")])
            task_domain_counter.update([sample.get("task_domain", "")])
            language_counter.update([sample.get("language", "")])
            if not sample.get("context", ""):
                empty_context += 1
            if not sample.get("reasoning", ""):
                empty_reasoning += 1

    return {
        "file": jsonl_path.name,
        "sample_count": total,
        "task_type_distribution": dict(task_type_counter),
        "task_domain_distribution": dict(task_domain_counter),
        "language_distribution": dict(language_counter),
        "empty_context_ratio": 0 if total == 0 else empty_context / total,
        "empty_reasoning_ratio": 0 if total == 0 else empty_reasoning / total,
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    summaries = [summarize_file(path) for path in sorted(input_dir.glob("*.jsonl"))]
    payload = {"files": summaries}
    if args.output_path:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
