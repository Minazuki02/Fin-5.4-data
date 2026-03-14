#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.distill.pipeline import DistillPipeline
from src.distill.provider import OpenAICompatibleProvider
from src.utils.logging_utils import configure_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the distillation runner."""
    parser = argparse.ArgumentParser(description="Run teacher-model distillation for normalized train/val samples.")
    parser.add_argument("--input_dir", default=str(REPO_ROOT / "data" / "normalized"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "distilled"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_concurrency", type=int, help="Override API concurrency from config.")
    parser.add_argument("--chunk_size", type=int, help="Override chunk size from config.")
    parser.add_argument("--prompt_version", help="Override prompt version from config.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load the distillation config YAML."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    """Run the end-to-end distillation pipeline."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    if args.max_concurrency is not None:
        config["api"]["max_concurrency"] = args.max_concurrency
    if args.chunk_size is not None:
        config["processing"]["chunk_size"] = args.chunk_size
    if args.prompt_version:
        config["generation"]["prompt_version"] = args.prompt_version

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_splits = list(config["processing"]["allowed_splits"])
    splits = args.splits or allowed_splits

    log_path = REPO_ROOT / "data" / "logs" / "distill.log"
    logger = configure_logger(log_path)
    provider = OpenAICompatibleProvider.from_config(config)
    pipeline = DistillPipeline(config=config, provider=provider, logger=logger)

    report = pipeline.process_all(
        input_dir=input_dir,
        output_dir=output_dir,
        sources=args.sources,
        splits=splits,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )

    stats_report_path = (REPO_ROOT / config["processing"]["stats_report_path"]).resolve()
    stats_report_path.parent.mkdir(parents=True, exist_ok=True)
    stats_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("distillation finished: %s", json.dumps(report, ensure_ascii=False))
    print(json.dumps({"stats_report_path": str(stats_report_path), "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
