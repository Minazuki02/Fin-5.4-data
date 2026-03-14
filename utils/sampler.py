from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SampleRef:
    source: str
    task_domain: str
    sample_id: str
    dataset_path: str
    line_number: int


@dataclass
class DatasetPool:
    source: str
    task_domain: str
    dataset_path: Path
    records: list[SampleRef]
    weight: float
    max_draws: int
    draws: int = 0

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def remaining_capacity(self) -> int:
        return self.max_draws - self.draws


def load_sampling_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid sampling config: {config_path}")
    return payload


def weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("weighted_choice received non-positive total weight")
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in items:
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1][0]


def discover_train_pools(normalized_root: Path, config: dict[str, Any]) -> dict[str, dict[str, DatasetPool]]:
    max_oversample_factor = float(config["max_oversample_factor"])
    if max_oversample_factor < 1.0:
        raise ValueError("max_oversample_factor must be >= 1.0")

    domain_weights: dict[str, float] = config["domain_weights"]
    dataset_weights: dict[str, dict[str, float]] = config["dataset_weights"]
    pools_by_domain: dict[str, dict[str, DatasetPool]] = {}

    for domain, domain_sources in dataset_weights.items():
        pools_by_domain[domain] = {}
        for source, weight in domain_sources.items():
            train_path = normalized_root / source / "train.jsonl"
            if not train_path.exists():
                raise FileNotFoundError(f"missing train split for {source}: {train_path}")

            records: list[SampleRef] = []
            with train_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    records.append(
                        SampleRef(
                            source=source,
                            task_domain=sample["task_domain"],
                            sample_id=sample["id"],
                            dataset_path=str(train_path),
                            line_number=line_number,
                        )
                    )

            if not records:
                raise ValueError(f"empty train split for {source}: {train_path}")

            pool = DatasetPool(
                source=source,
                task_domain=domain,
                dataset_path=train_path,
                records=records,
                weight=float(weight),
                max_draws=math.ceil(len(records) * max_oversample_factor),
            )
            pools_by_domain[domain][source] = pool

    return pools_by_domain


def total_capacity(pools_by_domain: dict[str, dict[str, DatasetPool]]) -> int:
    return sum(pool.max_draws for domain_pools in pools_by_domain.values() for pool in domain_pools.values())


def pool_statistics(pools_by_domain: dict[str, dict[str, DatasetPool]]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    total = 0
    for domain, domain_pools in pools_by_domain.items():
        datasets: dict[str, Any] = {}
        domain_total = 0
        for source, pool in domain_pools.items():
            datasets[source] = {
                "train_size": pool.size,
                "weight": pool.weight,
                "max_draws": pool.max_draws,
            }
            domain_total += pool.size
        by_domain[domain] = {
            "train_size": domain_total,
            "datasets": datasets,
        }
        total += domain_total
    return {"total_train_size": total, "domains": by_domain}


def sample_training_manifest(
    *,
    pools_by_domain: dict[str, dict[str, DatasetPool]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(config["seed"]))
    target_total_samples = int(config["target_total_samples"])
    domain_weights: dict[str, float] = {key: float(value) for key, value in config["domain_weights"].items()}

    capacity = total_capacity(pools_by_domain)
    if target_total_samples > capacity:
        raise ValueError(
            f"target_total_samples={target_total_samples} exceeds max sampling capacity={capacity}. "
            "Increase max_oversample_factor or reduce target_total_samples."
        )

    manifest: list[dict[str, Any]] = []
    domain_draws: dict[str, int] = {domain: 0 for domain in pools_by_domain}
    dataset_draws: dict[str, dict[str, int]] = {
        domain: {source: 0 for source in domain_pools}
        for domain, domain_pools in pools_by_domain.items()
    }

    for manifest_index in range(target_total_samples):
        eligible_domains: list[tuple[str, float]] = []
        for domain, weight in domain_weights.items():
            domain_pools = pools_by_domain.get(domain, {})
            if any(pool.remaining_capacity > 0 for pool in domain_pools.values()):
                eligible_domains.append((domain, weight))
        if not eligible_domains:
            raise RuntimeError("no eligible domains left while sampling manifest")

        selected_domain = weighted_choice(rng, eligible_domains)

        eligible_datasets: list[tuple[str, float]] = []
        for source, pool in pools_by_domain[selected_domain].items():
            if pool.remaining_capacity > 0:
                eligible_datasets.append((source, pool.weight))
        if not eligible_datasets:
            raise RuntimeError(f"no eligible datasets left in domain {selected_domain}")

        selected_source = weighted_choice(rng, eligible_datasets)
        pool = pools_by_domain[selected_domain][selected_source]
        sample_ref = pool.records[rng.randrange(pool.size)]
        pool.draws += 1
        domain_draws[selected_domain] += 1
        dataset_draws[selected_domain][selected_source] += 1

        manifest.append(
            {
                "manifest_index": manifest_index,
                "source": sample_ref.source,
                "task_domain": sample_ref.task_domain,
                "split": "train",
                "sample_id": sample_ref.sample_id,
                "dataset_path": sample_ref.dataset_path,
                "line_number": sample_ref.line_number,
                "draw_index_within_dataset": pool.draws,
            }
        )

    report = {
        "seed": int(config["seed"]),
        "target_total_samples": target_total_samples,
        "max_oversample_factor": float(config["max_oversample_factor"]),
        "domain_draws": domain_draws,
        "dataset_draws": dataset_draws,
        "dataset_oversampling": {
            domain: {
                source: {
                    "train_size": pool.size,
                    "draws": pool.draws,
                    "oversample_factor": pool.draws / pool.size,
                    "max_draws": pool.max_draws,
                }
                for source, pool in domain_pools.items()
            }
            for domain, domain_pools in pools_by_domain.items()
        },
    }
    return manifest, report


def write_manifest_jsonl(manifest: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

