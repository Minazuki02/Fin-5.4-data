#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.sampler import (
    discover_train_pools,
    load_sampling_config,
    pool_statistics,
    sample_training_manifest,
    total_capacity,
    write_manifest_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "sampling_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training manifest with hierarchical weighted sampling.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to sampling_config.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_sampling_config(config_path)

    normalized_root = (REPO_ROOT / config["normalized_root"]).resolve()
    output_manifest_path = (REPO_ROOT / config["output_manifest_path"]).resolve()
    output_report_path = (REPO_ROOT / config["output_report_path"]).resolve()

    pools_by_domain = discover_train_pools(normalized_root=normalized_root, config=config)
    stats = pool_statistics(pools_by_domain)
    manifest, sampling_report = sample_training_manifest(pools_by_domain=pools_by_domain, config=config)

    write_manifest_jsonl(manifest, output_manifest_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path),
        "normalized_root": str(normalized_root),
        "output_manifest_path": str(output_manifest_path),
        "pool_statistics": stats,
        "max_sampling_capacity": total_capacity(pools_by_domain),
        "sampling_report": sampling_report,
    }
    output_report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
