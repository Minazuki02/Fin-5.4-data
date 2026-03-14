#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a fixed number of JSONL records with a deterministic seed.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--sample_size", type=int, required=True, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--manifest", help="Optional JSON manifest path.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_sample_source_index"] = index
            records.append(record)
    return records


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.sample_size > len(records):
        raise ValueError(f"sample_size={args.sample_size} exceeds record_count={len(records)}")

    rng = random.Random(args.seed)
    chosen = rng.sample(records, args.sample_size)
    chosen.sort(key=lambda item: int(item["_sample_source_index"]))

    with output_path.open("w", encoding="utf-8") as handle:
        for record in chosen:
            record.pop("_sample_source_index", None)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "record_count": len(records),
            "sampled_ids": [str(record.get("id", "")) for record in chosen],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "seed": args.seed,
                "sample_size": args.sample_size,
                "record_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
