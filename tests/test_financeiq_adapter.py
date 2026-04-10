"""Unit tests for the financeiq adapter."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import pytest

from adapters.financeiq_adapter import SOURCE, TASK_TYPE, TASK_DOMAIN, LANGUAGE, build_records, normalize
from tests.conftest import assert_gold_answer_info, assert_normalized_sample, assert_split_report


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV_ROWS = [
    {
        "": "0",
        "Question": "以下哪项是资本市场的主要功能？",
        "A": "短期资金借贷",
        "B": "长期资金融通",
        "C": "外汇兑换",
        "D": "日常支付结算",
        "Answer": "B",
    },
    {
        "": "1",
        "Question": "企业发行股票属于什么融资方式？",
        "A": "间接融资",
        "B": "直接融资",
        "C": "内部融资",
        "D": "债务融资",
        "Answer": "B",
    },
    {
        "": "2",
        "Question": "下列哪个指标衡量企业盈利能力？",
        "A": "流动比率",
        "B": "资产负债率",
        "C": "净资产收益率",
        "D": "速动比率",
        "Answer": "C",
    },
]


def _write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows as a CSV file with headers from the first row."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests for build_records
# ---------------------------------------------------------------------------

class TestBuildRecords:
    """Tests for financeiq_adapter.build_records()."""

    def test_build_records_basic_structure(self, tmp_path: Path) -> None:
        """Each record must conform to the NormalizedSample schema."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "banking.csv", SAMPLE_CSV_ROWS)

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert len(records) == len(SAMPLE_CSV_ROWS)
        for rec in records:
            assert_normalized_sample(
                rec,
                source=SOURCE,
                task_type=TASK_TYPE,
                task_domain=TASK_DOMAIN,
                language=LANGUAGE,
            )

    def test_build_records_id_format(self, tmp_path: Path) -> None:
        """Record ids follow the pattern: financeiq_{split}_{stem}_{row_id}."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "test"
        _write_csv(raw_root / "markets.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["id"] == "financeiq_test_markets_0"

    def test_build_records_question_and_answer(self, tmp_path: Path) -> None:
        """Question text and reference_answer are correctly extracted."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS)

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["question"] == SAMPLE_CSV_ROWS[0]["Question"]
        assert records[0]["reference_answer"] == "B"
        assert records[2]["reference_answer"] == "C"

    def test_build_records_options_populated(self, tmp_path: Path) -> None:
        """MCQ options A-D are present in the expected 'X. text' format."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "quiz.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        options = records[0]["options"]
        assert len(options) == 4
        assert options[0] == "A. 短期资金借贷"
        assert options[1] == "B. 长期资金融通"
        assert options[2] == "C. 外汇兑换"
        assert options[3] == "D. 日常支付结算"

    def test_build_records_context_is_empty(self, tmp_path: Path) -> None:
        """financeiq samples should have empty context."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["context"] == ""

    def test_build_records_metadata_gold_answer_info(self, tmp_path: Path) -> None:
        """Metadata must include gold_answer_info with the correct structure."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        metadata = records[0]["metadata"]
        assert_gold_answer_info(metadata)
        gai = metadata["gold_answer_info"]
        assert gai["primary_answer"] == "B"
        assert gai["raw_answer_fields"]["Answer"] == "B"

    def test_build_records_metadata_raw_split(self, tmp_path: Path) -> None:
        """Metadata should store the raw_split directory name."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "validation"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["metadata"]["raw_split"] == "validation"

    def test_build_records_metadata_raw_sample(self, tmp_path: Path) -> None:
        """Metadata should include the raw CSV row."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        raw_sample = records[0]["metadata"]["raw_sample"]
        assert raw_sample["Question"] == SAMPLE_CSV_ROWS[0]["Question"]
        assert raw_sample["Answer"] == "B"

    def test_build_records_multiple_csv_files(self, tmp_path: Path) -> None:
        """Records from multiple CSV files in the same split are collected."""
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "banking.csv", SAMPLE_CSV_ROWS[:2])
        _write_csv(raw_root / "markets.csv", SAMPLE_CSV_ROWS[2:])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert len(records) == 3

    def test_build_records_multiple_splits(self, tmp_path: Path) -> None:
        """Records from different split directories are all collected."""
        for split_name in ("train", "dev"):
            raw_root = tmp_path / "FinanceIQ" / "data" / split_name
            _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS[:1])

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert len(records) == 2
        raw_splits = {rec["metadata"]["raw_split"] for rec in records}
        assert raw_splits == {"train", "dev"}

    def test_build_records_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty list when no CSV files exist."""
        (tmp_path / "FinanceIQ" / "data").mkdir(parents=True)

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records == []

    def test_build_records_row_without_id_uses_index(self, tmp_path: Path) -> None:
        """When both '' and 'id' keys are missing, row index is used."""
        rows = [
            {
                "Question": "测试问题",
                "A": "选项A",
                "B": "选项B",
                "C": "选项C",
                "D": "选项D",
                "Answer": "A",
            },
        ]
        raw_root = tmp_path / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", rows)

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["id"] == "financeiq_train_qa_0"


# ---------------------------------------------------------------------------
# Tests for normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    """Tests for financeiq_adapter.normalize()."""

    def test_normalize_writes_split_files(self, tmp_path: Path) -> None:
        """normalize() should produce train.jsonl, val.jsonl, test.jsonl."""
        raw_root = tmp_path / "raw" / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path / "raw"):
            report = normalize(output_dir=output_dir, seed=42)

        assert_split_report(report, SOURCE)
        for split_name in ("train", "val", "test"):
            split_path = output_dir / f"{split_name}.jsonl"
            assert split_path.exists(), f"{split_name}.jsonl was not created"

    def test_normalize_output_records_valid(self, tmp_path: Path) -> None:
        """Every record written to output files must be valid JSON and conform to schema."""
        raw_root = tmp_path / "raw" / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path / "raw"):
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

    def test_normalize_total_count_matches(self, tmp_path: Path) -> None:
        """The total across splits must equal the number of input rows."""
        raw_root = tmp_path / "raw" / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS)
        output_dir = tmp_path / "output"

        with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path / "raw"):
            report = normalize(output_dir=output_dir, seed=42)

        assert report["total_count"] == len(SAMPLE_CSV_ROWS)

    def test_normalize_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Running with the same seed produces identical output."""
        raw_root = tmp_path / "raw" / "FinanceIQ" / "data" / "train"
        _write_csv(raw_root / "qa.csv", SAMPLE_CSV_ROWS)

        results = []
        for i in range(2):
            output_dir = tmp_path / f"output_{i}"
            with mock.patch("adapters.financeiq_adapter.RAW_ROOT", tmp_path / "raw"):
                normalize(output_dir=output_dir, seed=42)
            split_records = {}
            for split_name in ("train", "val", "test"):
                with (output_dir / f"{split_name}.jsonl").open("r") as f:
                    split_records[split_name] = [json.loads(line) for line in f]
            results.append(split_records)

        for split_name in ("train", "val", "test"):
            assert results[0][split_name] == results[1][split_name]
