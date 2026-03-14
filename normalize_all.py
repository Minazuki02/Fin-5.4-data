#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters import ALL_ADAPTERS
from adapters.common import NORMALIZED_ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize all financial datasets into unified jsonl files.")
    parser.add_argument(
        "--output-dir",
        default=str(NORMALIZED_ROOT),
        help="Directory for normalized jsonl files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible re-splitting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    summary: list[dict[str, object]] = []
    for adapter in ALL_ADAPTERS:
        source_name = adapter.__module__.split(".")[-1].replace("_adapter", "")
        dataset_output_dir = output_dir / source_name
        summary.append(adapter(dataset_output_dir, seed=args.seed))

    payload = {
        "seed": args.seed,
        "datasets": summary,
    }
    report_path = output_dir / "split_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split_report_path": str(report_path), "datasets": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
