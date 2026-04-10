"""Unit tests for the finqa adapter."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from adapters.finqa_adapter import SOURCE, TASK_TYPE, TASK_DOMAIN, LANGUAGE, build_records, normalize
from tests.conftest import assert_gold_answer_info, assert_normalized_sample, assert_split_report


# ---------------------------------------------------------------------------
# Sample data matching the FinQA raw JSON format
# ---------------------------------------------------------------------------

SAMPLE_FINQA_RECORDS = [
    {
        "id": "finqa_train_0",
        "pre_text": [
            "the following table shows the revenue breakdown by segment.",
            "all amounts are in millions of dollars.",
        ],
        "table": [
            ["", "2019", "2018", "2017"],
            ["Revenue", "$1,200", "$1,100", "$1,000"],
            ["Cost", "$800", "$750", "$700"],
            ["Profit", "$400", "$350", "$300"],
        ],
        "post_text": [
            "the company achieved steady growth over the three-year period.",
        ],
        "qa": {
            "question": "What was the percentage increase in revenue from 2017 to 2019?",
            "answer": "20%",
            "exe_ans": 20.0,
            "final_result": "20%",
        },
    },
    {
        "id": "finqa_train_1",
        "pre_text": ["net income for the fiscal year was as follows."],
        "table": [
            ["Year", "Net Income"],
            ["2020", "$500M"],
            ["2019", "$450M"],
        ],
        "post_text": [],
        "qa": {
            "question": "What is the increase in net income from 2019 to 2020?",
            "answer": "$50M",
            "exe_ans": 50.0,
        },
    },
    {
        "id": "finqa_train_2",
        "pre_text": [],
        "table": [
            ["Metric", "Value"],
            ["EPS", "$3.50"],
        ],
        "post_text": ["earnings per share increased year over year."],
        "qa": {
            "question": "What was the EPS?",
            "answer": "$3.50",
        },
    },
]


def _write_finqa_json(json_path: Path, records: list[dict]) -> None:
    """Write sample records in the FinQA raw JSON format."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _make_finqa_raw_root(tmp_path: Path) -> Path:
    """Set up a mock FinQA-main directory with train/dev/test splits."""
    root = tmp_path / "finqa" / "FinQA-main"
    dataset_dir = root / "dataset"
    _write_finqa_json(dataset_dir / "train.json", SAMPLE_FINQA_RECORDS)
    _write_finqa_json(dataset_dir / "dev.json", SAMPLE_FINQA_RECORDS[:1])
    _write_finqa_json(dataset_dir / "test.json", SAMPLE_FINQA_RECORDS[:1])
    return root


# ---------------------------------------------------------------------------
# Tests for build_records
# ---------------------------------------------------------------------------

class TestBuildRecords:
    """Tests for finqa_adapter.build_records()."""

    def test_build_records_basic_structure(self, tmp_path: Path) -> None:
        """Each record must conform to the NormalizedSample schema."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        assert len(records) == 5  # 3 train + 1 dev + 1 test
        for rec in records:
            assert_normalized_sample(
                rec,
                source=SOURCE,
                task_type=TASK_TYPE,
                task_domain=TASK_DOMAIN,
                language=LANGUAGE,
            )

    def test_build_records_id_preserved(self, tmp_path: Path) -> None:
        """Record ids should come from the raw sample 'id' field."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        ids = [rec["id"] for rec in records]
        assert "finqa_train_0" in ids
        assert "finqa_train_1" in ids

    def test_build_records_question_extracted(self, tmp_path: Path) -> None:
        """The question field should come from qa.question."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        train_records = [r for r in records if r["metadata"]["raw_split"] == "train"]
        assert train_records[0]["question"] == "What was the percentage increase in revenue from 2017 to 2019?"

    def test_build_records_answer_resolution(self, tmp_path: Path) -> None:
        """reference_answer should be resolved via first_non_empty(answer, final_result, exe_ans)."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        train_records = [r for r in records if r["metadata"]["raw_split"] == "train"]
        # First record has answer="20%"
        assert train_records[0]["reference_answer"] == "20%"
        # Second record has answer="$50M"
        assert train_records[1]["reference_answer"] == "$50M"
        # Third record has answer="$3.50"
        assert train_records[2]["reference_answer"] == "$3.50"

    def test_build_records_context_includes_table(self, tmp_path: Path) -> None:
        """Context should include pre_text, table, and post_text joined."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        train_records = [r for r in records if r["metadata"]["raw_split"] == "train"]
        ctx = train_records[0]["context"]
        # pre_text content
        assert "revenue breakdown by segment" in ctx
        # table content
        assert "Revenue" in ctx
        assert "$1,200" in ctx
        # post_text content
        assert "steady growth" in ctx

    def test_build_records_options_empty(self, tmp_path: Path) -> None:
        """FinQA is a reasoning task, so options should be empty."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        for rec in records:
            assert rec["options"] == []

    def test_build_records_metadata_gold_answer_info(self, tmp_path: Path) -> None:
        """Metadata must contain gold_answer_info with correct structure."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        for rec in records:
            assert_gold_answer_info(rec["metadata"])

        train_records = [r for r in records if r["metadata"]["raw_split"] == "train"]
        gai = train_records[0]["metadata"]["gold_answer_info"]
        assert gai["primary_answer"] == "20%"
        assert gai["execution_answer"] == "20.0"
        assert gai["raw_answer_fields"]["answer"] == "20%"
        assert gai["raw_answer_fields"]["final_result"] == "20%"

    def test_build_records_metadata_raw_split(self, tmp_path: Path) -> None:
        """Metadata raw_split should reflect the source file split name."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        raw_splits = {rec["metadata"]["raw_split"] for rec in records}
        assert raw_splits == {"train", "dev", "test"}

    def test_build_records_metadata_raw_sample(self, tmp_path: Path) -> None:
        """Metadata should preserve the full raw sample dict."""
        raw_root = _make_finqa_raw_root(tmp_path)

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        train_records = [r for r in records if r["metadata"]["raw_split"] == "train"]
        raw = train_records[0]["metadata"]["raw_sample"]
        assert raw["id"] == "finqa_train_0"
        assert "qa" in raw
        assert "pre_text" in raw

    def test_build_records_fallback_id(self, tmp_path: Path) -> None:
        """When 'id' field is missing, a generated id should be used."""
        no_id_record = {
            "pre_text": ["test"],
            "table": [],
            "post_text": [],
            "qa": {
                "question": "Test?",
                "answer": "42",
            },
        }
        root = tmp_path / "finqa" / "FinQA-main" / "dataset"
        _write_finqa_json(root / "train.json", [no_id_record])
        _write_finqa_json(root / "dev.json", [])
        _write_finqa_json(root / "test.json", [])

        raw_root = tmp_path / "finqa" / "FinQA-main"
        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        assert records[0]["id"] == "finqa_train_0"

    def test_build_records_answer_fallback_to_exe_ans(self, tmp_path: Path) -> None:
        """When 'answer' is missing, fall back to exe_ans."""
        record_no_answer = {
            "id": "test_fallback",
            "pre_text": [],
            "table": [],
            "post_text": [],
            "qa": {
                "question": "Fallback test?",
                "exe_ans": 99.5,
            },
        }
        root = tmp_path / "finqa" / "FinQA-main" / "dataset"
        _write_finqa_json(root / "train.json", [record_no_answer])
        _write_finqa_json(root / "dev.json", [])
        _write_finqa_json(root / "test.json", [])

        raw_root = tmp_path / "finqa" / "FinQA-main"
        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            records = build_records()

        assert records[0]["reference_answer"] == "99.5"


# ---------------------------------------------------------------------------
# Tests for normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    """Tests for finqa_adapter.normalize()."""

    def test_normalize_writes_split_files(self, tmp_path: Path) -> None:
        """normalize() should produce train.jsonl, val.jsonl, test.jsonl."""
        raw_root = _make_finqa_raw_root(tmp_path)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            report = normalize(output_dir=output_dir, seed=42)

        assert_split_report(report, SOURCE)
        for split_name in ("train", "val", "test"):
            split_path = output_dir / f"{split_name}.jsonl"
            assert split_path.exists(), f"{split_name}.jsonl was not created"

    def test_normalize_output_records_valid(self, tmp_path: Path) -> None:
        """Every record in the output files must conform to the schema."""
        raw_root = _make_finqa_raw_root(tmp_path)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            normalize(output_dir=output_dir, seed=42)

        for split_name in ("train", "val", "test"):
            split_path = output_dir / f"{split_name}.jsonl"
            with split_path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    assert_normalized_sample(
                        record,
                        source=SOURCE,
                        task_type=TASK_TYPE,
                        task_domain=TASK_DOMAIN,
                        language=LANGUAGE,
                    )
                    assert record["split"] == split_name

    def test_normalize_preserves_original_splits(self, tmp_path: Path) -> None:
        """When train/dev/test all exist, original splits should be preserved."""
        raw_root = _make_finqa_raw_root(tmp_path)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
            report = normalize(output_dir=output_dir, seed=42)

        assert report["total_count"] == 5

    def test_normalize_deterministic(self, tmp_path: Path) -> None:
        """Running normalize twice with the same seed yields identical output."""
        raw_root = _make_finqa_raw_root(tmp_path)

        results = []
        for i in range(2):
            output_dir = tmp_path / f"output_{i}"
            with mock.patch("adapters.finqa_adapter.dataset_root", return_value=raw_root):
                normalize(output_dir=output_dir, seed=42)
            split_records = {}
            for split_name in ("train", "val", "test"):
                with (output_dir / f"{split_name}.jsonl").open("r") as f:
                    split_records[split_name] = [json.loads(line) for line in f]
            results.append(split_records)

        for split_name in ("train", "val", "test"):
            assert results[0][split_name] == results[1][split_name]
