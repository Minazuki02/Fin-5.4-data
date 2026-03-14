#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_parent, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish final FinancialPhraseBank filtered data into train_data.")
    parser.add_argument("--round1_keep", nargs=2, required=True)
    parser.add_argument("--round1_drop", nargs=2, required=True)
    parser.add_argument("--round2_keep", nargs=2, required=True)
    parser.add_argument("--round2_drop", nargs=2, required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "train_data" / "financial_phrasebank"))
    return parser.parse_args()


def write_copy(src: Path, dst: Path) -> int:
    ensure_parent(dst)
    count = 0
    with dst.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(src):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    dropped_dir = output_dir / "dropped"

    counts = {
        "train_final_keep": write_copy(Path(args.round2_keep[0]).resolve(), output_dir / "train.jsonl"),
        "val_final_keep": write_copy(Path(args.round2_keep[1]).resolve(), output_dir / "val.jsonl"),
        "train_round1_drop": write_copy(Path(args.round1_drop[0]).resolve(), dropped_dir / "train.round1.drop.jsonl"),
        "val_round1_drop": write_copy(Path(args.round1_drop[1]).resolve(), dropped_dir / "val.round1.drop.jsonl"),
        "train_round2_drop": write_copy(Path(args.round2_drop[0]).resolve(), dropped_dir / "train.round2.drop.jsonl"),
        "val_round2_drop": write_copy(Path(args.round2_drop[1]).resolve(), dropped_dir / "val.round2.drop.jsonl"),
        "train_round1_keep_snapshot": write_copy(Path(args.round1_keep[0]).resolve(), dropped_dir / "train.round1.keep_snapshot.jsonl"),
        "val_round1_keep_snapshot": write_copy(Path(args.round1_keep[1]).resolve(), dropped_dir / "val.round1.keep_snapshot.jsonl"),
    }

    summary = {"output_dir": str(output_dir), **counts}
    summary_path = output_dir / "publish_summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
