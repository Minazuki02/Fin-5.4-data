from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.execution_eval import evaluate_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for execution evaluation."""
    parser = argparse.ArgumentParser(description="Compute execution accuracy for FinQA or ConvFinQA predictions.")
    parser.add_argument("--input", required=True, help="Path to a prediction JSONL file.")
    return parser.parse_args()


def main() -> int:
    """Run minimal execution evaluation and print the metrics as JSON."""
    args = parse_args()
    result = evaluate_file(Path(args.input).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
