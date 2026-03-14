from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from adapters.common import ensure_dir, write_jsonl


TRAIN_SPLIT_NAMES = {"train", "training"}
VAL_SPLIT_NAMES = {"val", "dev", "valid", "validation", "eval"}
TEST_SPLIT_NAMES = {"test", "test_gold"}
OUTPUT_SPLITS = ["train", "val", "test"]
DEFAULT_SEED = 42
DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def canonicalize_split_name(raw_split: Any) -> str:
    text = str(raw_split or "").strip().lower()
    if text in TRAIN_SPLIT_NAMES:
        return "train"
    if text in VAL_SPLIT_NAMES:
        return "val"
    if text in TEST_SPLIT_NAMES:
        return "test"
    return ""


def extract_raw_split(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("raw_split", "") or "").strip()


def summarize_original_splits(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    raw_counter: Counter[str] = Counter()
    canonical_counter: Counter[str] = Counter()

    for record in records:
        raw_split = extract_raw_split(record)
        canonical_split = canonicalize_split_name(raw_split)
        raw_counter.update([raw_split or "__missing__"])
        canonical_counter.update([canonical_split or "__missing__"])

    return dict(raw_counter), dict(canonical_counter)


def has_complete_original_split(records: list[dict[str, Any]]) -> bool:
    canonical_splits = {canonicalize_split_name(extract_raw_split(record)) for record in records}
    if "" in canonical_splits:
        return False
    return canonical_splits == {"train", "val", "test"}


def calculate_split_sizes(total_count: int, ratios: dict[str, float]) -> dict[str, int]:
    if total_count < 3:
        raise ValueError(f"dataset is too small to create non-empty train/val/test splits: total_count={total_count}")

    counts = {split: 1 for split in OUTPUT_SPLITS}
    remaining = total_count - len(OUTPUT_SPLITS)

    extra_counts: dict[str, int] = {}
    fractional_parts: list[tuple[float, str]] = []
    for split in OUTPUT_SPLITS:
        raw_extra = remaining * ratios[split]
        integer_extra = int(raw_extra)
        extra_counts[split] = integer_extra
        fractional_parts.append((raw_extra - integer_extra, split))

    for split in OUTPUT_SPLITS:
        counts[split] += extra_counts[split]

    leftover = total_count - sum(counts.values())
    for _, split in sorted(fractional_parts, key=lambda item: (-item[0], OUTPUT_SPLITS.index(item[1]))):
        if leftover <= 0:
            break
        counts[split] += 1
        leftover -= 1

    return counts


def normalize_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    normalized = dict(record)
    metadata = dict(record.get("metadata", {}))
    metadata["raw_split"] = extract_raw_split(record)
    normalized["metadata"] = metadata
    normalized["split"] = split
    return normalized


def assign_preserved_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {split: [] for split in OUTPUT_SPLITS}
    for record in records:
        split = canonicalize_split_name(extract_raw_split(record))
        grouped[split].append(normalize_record(record, split))
    return grouped


def assign_resplit_records(records: list[dict[str, Any]], seed: int, ratios: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    counts = calculate_split_sizes(len(records), ratios)
    shuffled = [normalize_record(record, "") for record in sorted(records, key=lambda item: str(item.get("id", "")))]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    grouped = {split: [] for split in OUTPUT_SPLITS}
    start = 0
    for split in OUTPUT_SPLITS:
        end = start + counts[split]
        for record in shuffled[start:end]:
            record["split"] = split
            grouped[split].append(record)
        start = end
    return grouped


def write_split_files(grouped_records: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, str]:
    ensure_dir(output_dir)
    output_paths: dict[str, str] = {}
    for split in OUTPUT_SPLITS:
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(grouped_records[split], output_path)
        output_paths[split] = str(output_path)
    return output_paths


def split_and_write_dataset(
    *,
    source: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    effective_ratios = ratios or DEFAULT_RATIOS
    raw_distribution, canonical_distribution = summarize_original_splits(records)
    preserve_original = has_complete_original_split(records)

    if preserve_original:
        grouped_records = assign_preserved_splits(records)
    else:
        grouped_records = assign_resplit_records(records, seed=seed, ratios=effective_ratios)

    output_paths = write_split_files(grouped_records, output_dir)
    counts = {split: len(grouped_records[split]) for split in OUTPUT_SPLITS}

    return {
        "source": source,
        "original_split_distribution": raw_distribution,
        "canonical_split_distribution": canonical_distribution,
        "preserve_original_split": preserve_original,
        "re_split": not preserve_original,
        "split_ratio": dict(effective_ratios),
        "seed": seed,
        "counts": counts,
        "output_paths": output_paths,
        "total_count": sum(counts.values()),
    }
