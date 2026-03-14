#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DEFAULT_MANIFEST_PATH = MANIFEST_DIR / "download_manifest.json"


@dataclass(frozen=True)
class DatasetSpec:
    source_name: str
    provider: str
    identifier: str
    url: str
    notes: str = ""


DATASETS: list[DatasetSpec] = [
    DatasetSpec("Duxiaoman-DI_FinCorpus", "huggingface", "Duxiaoman-DI/FinCorpus", "https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus"),
    DatasetSpec("Duxiaoman-DI_FinanceIQ", "huggingface", "Duxiaoman-DI/FinanceIQ", "https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ"),
    DatasetSpec("Macielyoung_FinCUGE_Instruction", "git", "https://github.com/Macielyoung/FinCUGE_Instruction.git", "https://github.com/Macielyoung/FinCUGE_Instruction.git"),
    DatasetSpec("Shanghai_AI_Laboratory_open-compass-OpenFinData", "modelscope", "Shanghai_AI_Laboratory/open-compass-OpenFinData", "https://modelscope.cn/datasets/Shanghai_AI_Laboratory/open-compass-OpenFinData"),
    DatasetSpec("ibm-research_finqa", "huggingface", "ibm-research/finqa", "https://huggingface.co/datasets/ibm-research/finqa"),
    DatasetSpec("AdaptLLM_ConvFinQA", "huggingface", "AdaptLLM/ConvFinQA", "https://huggingface.co/datasets/AdaptLLM/ConvFinQA"),
    DatasetSpec("next-tat_TAT-QA", "huggingface", "next-tat/TAT-QA", "https://huggingface.co/datasets/next-tat/TAT-QA"),
    DatasetSpec("gbharti_finance-alpaca", "huggingface", "gbharti/finance-alpaca", "https://huggingface.co/datasets/gbharti/finance-alpaca"),
    DatasetSpec("antgroup_Agentar-DeepFinance-100K", "git", "https://github.com/antgroup/Agentar-DeepFinance-100K.git", "https://github.com/antgroup/Agentar-DeepFinance-100K.git"),
    DatasetSpec("zeroshot_twitter-financial-news-sentiment", "huggingface", "zeroshot/twitter-financial-news-sentiment", "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment"),
    DatasetSpec("ChanceFocus_fiqa-sentiment-classification", "huggingface", "ChanceFocus/fiqa-sentiment-classification", "https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification"),
    DatasetSpec("takala_financial_phrasebank", "huggingface", "takala/financial_phrasebank", "https://huggingface.co/datasets/takala/financial_phrasebank"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_files_and_size(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    if not path.exists():
        return file_count, total_bytes

    for file_path in path.rglob("*"):
        if file_path.is_file():
            file_count += 1
            total_bytes += file_path.stat().st_size
    return file_count, total_bytes


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )


def format_command_error(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or "unknown command failure"
    return f"exit_code={result.returncode}: {details}"


def python_executable() -> str:
    return sys.executable or "python"


def has_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def download_from_huggingface(spec: DatasetSpec, target_dir: Path) -> None:
    if not has_module("huggingface_hub"):
        raise RuntimeError("missing dependency: huggingface_hub")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec.identifier,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_from_modelscope(spec: DatasetSpec, target_dir: Path) -> None:
    if not has_module("modelscope"):
        raise RuntimeError("missing dependency: modelscope")

    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        spec.identifier,
        repo_type="dataset",
        local_dir=str(target_dir),
    )


def download_from_git(spec: DatasetSpec, target_dir: Path) -> None:
    git_dir = target_dir / ".git"
    if git_dir.exists():
        result = run_command(["git", "-C", str(target_dir), "pull", "--ff-only"])
        if result.returncode != 0:
            download_github_archive(spec, target_dir)
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    result = run_command(["git", "clone", spec.identifier, str(target_dir)])
    if result.returncode != 0:
        download_github_archive(spec, target_dir)


def download_github_archive(spec: DatasetSpec, target_dir: Path) -> None:
    if "github.com" not in spec.url:
        raise RuntimeError("git clone failed and no GitHub archive fallback is available")

    repo_url = spec.url.removesuffix(".git").rstrip("/")
    archive_urls = [
        f"{repo_url}/archive/refs/heads/main.zip",
        f"{repo_url}/archive/refs/heads/master.zip",
    ]

    temp_zip = target_dir.parent / f"{target_dir.name}.zip"
    temp_extract = target_dir.parent / f"{target_dir.name}__extract"

    if temp_zip.exists():
        temp_zip.unlink()
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    last_error: str | None = None
    for archive_url in archive_urls:
        try:
            urllib.request.urlretrieve(archive_url, temp_zip)
            ensure_directory(temp_extract)
            with zipfile.ZipFile(temp_zip, "r") as zip_file:
                zip_file.extractall(temp_extract)

            extracted_roots = [item for item in temp_extract.iterdir() if item.is_dir()]
            if len(extracted_roots) != 1:
                raise RuntimeError("unexpected GitHub archive layout")

            extracted_root = extracted_roots[0]
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(extracted_root), str(target_dir))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        finally:
            if temp_zip.exists():
                temp_zip.unlink()
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

    raise RuntimeError(f"git clone failed and GitHub archive fallback failed: {last_error}")


def download_dataset(spec: DatasetSpec, target_dir: Path) -> None:
    if spec.provider == "huggingface":
        download_from_huggingface(spec, target_dir)
        return
    if spec.provider == "modelscope":
        download_from_modelscope(spec, target_dir)
        return
    if spec.provider == "git":
        download_from_git(spec, target_dir)
        return
    raise ValueError(f"unsupported provider: {spec.provider}")


def build_manifest_record(spec: DatasetSpec, target_dir: Path, success: bool, started_at: str, finished_at: str, error: str | None) -> dict[str, Any]:
    file_count, total_bytes = count_files_and_size(target_dir)
    return {
        "source_name": spec.source_name,
        "provider": spec.provider,
        "identifier": spec.identifier,
        "url": spec.url,
        "local_path": str(target_dir),
        "download_success": success,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "notes": spec.notes,
    }


def write_manifest(records: list[dict[str, Any]], manifest_path: Path) -> None:
    ensure_directory(manifest_path.parent)
    payload = {
        "generated_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "raw_root": str(RAW_ROOT),
        "dataset_count": len(records),
        "success_count": sum(1 for record in records if record["download_success"]),
        "failure_count": sum(1 for record in records if not record["download_success"]),
        "datasets": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required open financial datasets into data/raw.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to write the manifest JSON.")
    parser.add_argument("--force", action="store_true", help="Delete existing target directories before download.")
    parser.add_argument("--sources", nargs="*", help="Optional subset of source_name values to download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directory(RAW_ROOT)
    manifest_path = Path(args.manifest).resolve()

    selected = {name for name in args.sources} if args.sources else None
    specs = [spec for spec in DATASETS if selected is None or spec.source_name in selected]

    records: list[dict[str, Any]] = []
    for spec in specs:
        target_dir = RAW_ROOT / spec.source_name
        started_at = utc_now()
        error: str | None = None
        success = False

        try:
            if args.force and target_dir.exists():
                shutil.rmtree(target_dir)
            ensure_directory(target_dir.parent)
            download_dataset(spec, target_dir)
            success = True
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            finished_at = utc_now()
            records.append(build_manifest_record(spec, target_dir, success, started_at, finished_at, error))
            write_manifest(records, manifest_path)

    print(json.dumps({"manifest": str(manifest_path), "datasets": len(records)}, ensure_ascii=False))
    return 0 if all(record["download_success"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
