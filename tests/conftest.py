"""Shared fixtures for adapter unit tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


NORMALIZED_FIELDS = [
    "id",
    "source",
    "split",
    "task_type",
    "task_domain",
    "language",
    "question",
    "context",
    "options",
    "reference_answer",
    "metadata",
]


def assert_normalized_sample(
    record: dict[str, Any],
    *,
    source: str,
    task_type: str,
    task_domain: str,
    language: str,
) -> None:
    """Validate that a record conforms to the NormalizedSample schema."""
    for field in NORMALIZED_FIELDS:
        assert field in record, f"Missing field: {field}"

    assert isinstance(record["id"], str) and record["id"], "id must be a non-empty string"
    assert record["source"] == source
    assert record["split"] in ("train", "val", "test", "")
    assert record["task_type"] == task_type
    assert record["task_domain"] == task_domain
    assert record["language"] == language
    assert isinstance(record["question"], str)
    assert isinstance(record["context"], str)
    assert isinstance(record["options"], list)
    assert isinstance(record["reference_answer"], str)
    assert isinstance(record["metadata"], dict)


def assert_gold_answer_info(metadata: dict[str, Any]) -> None:
    """Validate the gold_answer_info sub-structure in metadata."""
    gai = metadata.get("gold_answer_info")
    assert gai is not None, "metadata must contain gold_answer_info"
    assert isinstance(gai["primary_answer"], str)
    assert isinstance(gai["intermediate_answers"], list)
    assert isinstance(gai["execution_answer"], str)
    assert isinstance(gai["answer_type"], str)
    assert isinstance(gai["answer_from"], str)
    assert isinstance(gai["raw_answer_fields"], dict)


def assert_split_report(report: dict[str, Any], source: str) -> None:
    """Validate the report dict returned by normalize()."""
    assert report["source"] == source
    assert "counts" in report
    assert "total_count" in report
    assert report["total_count"] > 0
    for split_name in ("train", "val", "test"):
        assert split_name in report["counts"]
        assert report["counts"][split_name] >= 0
    assert sum(report["counts"].values()) == report["total_count"]
    assert "output_paths" in report
