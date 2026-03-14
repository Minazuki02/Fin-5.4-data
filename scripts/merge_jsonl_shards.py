#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_parent


SHARD_PATTERN = re.compile(r"^(?P<stem>.+)\.(?P<start>\d+)_(?P<end>\d+)(?P<suffix>\.[^.]+(?:\.[^.]+)?)$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for merging sharded JSONL files."""
    parser = argparse.ArgumentParser(description="Merge JSONL shard files in numeric shard order.")
    parser.add_argument("--input_dir", required=True, help="Directory containing shard files.")
    parser.add_argument("--pattern", required=True, help="Glob pattern for shard files, e.g. train.*_*.keep.jsonl")
    parser.add_argument("--output", required=True, help="Merged JSONL output path.")
    return parser.parse_args()


def shard_key(path: Path) -> tuple[int, int, str]:
    """Return a stable sort key from a shard filename."""
    match = SHARD_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"not a shard-style file name: {path.name}")
    return (int(match.group("start")), int(match.group("end")), path.name)


def main() -> int:
    """Merge shard files to one output JSONL."""
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    shard_paths = sorted(input_dir.glob(args.pattern), key=shard_key)

    ensure_parent(output_path)
    merged_count = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8-sig") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line)
                        merged_count += 1

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "pattern": args.pattern,
                "output": str(output_path),
                "shard_count": len(shard_paths),
                "merged_count": merged_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
