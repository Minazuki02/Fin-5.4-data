#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "manifest.json"


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


def verify_record(record: dict[str, Any]) -> dict[str, Any]:
    local_path = Path(record["local_path"])
    exists = local_path.exists()
    file_count, total_size_bytes = count_files_and_size(local_path)

    issues: list[str] = []
    if not exists:
        issues.append("missing_path")
    if exists and file_count == 0:
        issues.append("empty_directory")
    if record.get("file_count") != file_count:
        issues.append("file_count_mismatch")
    if record.get("total_size_bytes") != total_size_bytes:
        issues.append("total_size_bytes_mismatch")
    if record.get("success") is True and issues:
        issues.append("manifest_success_but_verification_failed")
    if record.get("success") is False and exists and file_count > 0:
        issues.append("manifest_failed_but_files_exist")

    return {
        "source_name": record["source_name"],
        "repo_id": record["repo_id"],
        "local_path": str(local_path),
        "manifest_success": record.get("success", False),
        "exists": exists,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "verified": len(issues) == 0,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify downloaded dataset directories against manifest.json.")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_records = payload.get("datasets", [])
    results = [verify_record(record) for record in dataset_records]

    summary = {
        "manifest_path": str(manifest_path),
        "dataset_count": len(results),
        "verified_count": sum(1 for result in results if result["verified"]),
        "failed_count": sum(1 for result in results if not result["verified"]),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
