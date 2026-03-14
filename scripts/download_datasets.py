#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "manifest.json"


@dataclass(frozen=True)
class DatasetSpec:
    source_name: str
    repo_id: str


DATASETS: list[DatasetSpec] = [
    DatasetSpec(source_name="FinanceIQ", repo_id="Duxiaoman-DI/FinanceIQ"),
    DatasetSpec(source_name="FinEval", repo_id="SUFE-AIFLM-Lab/FinEval"),
    DatasetSpec(source_name="finqa", repo_id="ibm-research/finqa"),
    DatasetSpec(source_name="ConvFinQA", repo_id="AdaptLLM/ConvFinQA"),
    DatasetSpec(source_name="TAT-QA", repo_id="next-tat/TAT-QA"),
    DatasetSpec(source_name="FinCUGE-Instruction", repo_id="Maciel/FinCUGE-Instruction"),
    DatasetSpec(source_name="finance-alpaca", repo_id="gbharti/finance-alpaca"),
    DatasetSpec(source_name="financial_phrasebank", repo_id="takala/financial_phrasebank"),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_files_and_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    file_count = 0
    total_size_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_size_bytes += item.stat().st_size
    return file_count, total_size_bytes


def write_manifest(records: list[dict[str, Any]], manifest_path: Path) -> None:
    ensure_dir(manifest_path.parent)
    payload = {
        "dataset_count": len(records),
        "datasets": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_dataset(spec: DatasetSpec, target_dir: Path, token: str | None) -> tuple[bool, str | None]:
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            token=token,
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def build_record(spec: DatasetSpec, target_dir: Path, success: bool, error_message: str | None) -> dict[str, Any]:
    file_count, total_size_bytes = count_files_and_size(target_dir)
    return {
        "source_name": spec.source_name,
        "repo_id": spec.repo_id,
        "local_path": str(target_dir),
        "success": success,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "error_message": error_message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fixed Hugging Face datasets into data/raw.")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to write manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing dataset directories before downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    manifest_path = Path(args.manifest_path).resolve()

    ensure_dir(RAW_ROOT)
    records: list[dict[str, Any]] = []

    for spec in DATASETS:
        target_dir = RAW_ROOT / spec.source_name

        if args.force and target_dir.exists():
            shutil.rmtree(target_dir)

        ensure_dir(target_dir.parent)
        success, error_message = download_dataset(spec, target_dir, token)
        records.append(build_record(spec, target_dir, success, error_message))
        write_manifest(records, manifest_path)

    all_success = all(record["success"] for record in records)
    print(json.dumps({"manifest_path": str(manifest_path), "all_success": all_success}, ensure_ascii=False))
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
