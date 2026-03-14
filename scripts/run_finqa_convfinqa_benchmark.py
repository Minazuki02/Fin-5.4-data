from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.llm_benchmark import (  # noqa: E402
    DEFAULT_DATASET_PATHS,
    OpenAIResponsesProvider,
    benchmark_output_dir,
    run_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gpt-5.4 on FinQA and ConvFinQA test sets via the Responses API."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASET_PATHS.keys()),
        default=["finqa", "convfinqa"],
        help="One or more benchmark datasets to evaluate.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "distill.yaml"),
        help="Config file used to read API env names and timeout defaults.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for prediction files and summary.json. Defaults to a timestamped folder under data/benchmarks.",
    )
    parser.add_argument("--model_name", default="gpt-5.4", help="Model name to send to the Responses API.")
    parser.add_argument(
        "--reasoning_effort",
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Responses API reasoning effort.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=1200,
        help="Maximum output tokens per sample.",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent API calls.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Optional cap per dataset for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing prediction files and regenerate everything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_paths = {name: DEFAULT_DATASET_PATHS[name] for name in args.datasets}
    missing = [str(path) for path in dataset_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else benchmark_output_dir(REPO_ROOT / "data" / "benchmarks", args.model_name)
    )
    provider = OpenAIResponsesProvider.from_config(
        config_path=Path(args.config).resolve(),
        model_name=args.model_name,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
    )
    summary = run_benchmark(
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        provider=provider,
        max_samples=args.max_samples,
        max_concurrency=args.max_concurrency,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
