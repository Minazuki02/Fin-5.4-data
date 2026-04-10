"""Unit tests for the convfinqa adapter."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from adapters.convfinqa_adapter import (
    SOURCE,
    TASK_TYPE,
    TASK_DOMAIN,
    LANGUAGE,
    turn_index_from_id,
    active_turn,
    history_to_text,
    turn_number,
    convfinqa_gold_answer_info,
    build_records,
    normalize,
)
from tests.conftest import assert_gold_answer_info, assert_normalized_sample, assert_split_report


# ---------------------------------------------------------------------------
# Sample data matching the ConvFinQA raw JSON format
# ---------------------------------------------------------------------------

SAMPLE_SINGLE_TURN = {
    "id": "convfinqa_single_0",
    "pre_text": ["the following shows quarterly results."],
    "table": [
        ["Quarter", "Revenue", "Expenses"],
        ["Q1", "$100M", "$60M"],
        ["Q2", "$120M", "$70M"],
    ],
    "post_text": ["revenue increased quarter over quarter."],
    "qa": {
        "question": "What was the revenue increase from Q1 to Q2?",
        "answer": "$20M",
        "exe_ans": 20.0,
    },
}

SAMPLE_MULTI_TURN = {
    "id": "convfinqa_multi_1",
    "pre_text": ["annual report summary for fiscal year 2020."],
    "table_ori": [
        ["Year", "Revenue", "Net Income"],
        ["2020", "$500M", "$80M"],
        ["2019", "$450M", "$70M"],
    ],
    "post_text": [],
    "qa_0": {
        "question": "What was the revenue in 2020?",
        "answer": "$500M",
        "exe_ans": 500.0,
    },
    "qa_1": {
        "question": "What was the increase in net income from 2019 to 2020?",
        "answer": "$10M",
        "exe_ans": 10.0,
        "answer_list": ["$80M", "$70M", "$10M"],
    },
    "annotation": {
        "answer_list": ["$500M"],
        "answer_list_0": ["$500M"],
        "answer_list_1": ["$80M", "$70M", "$10M"],
    },
}

SAMPLE_MULTI_TURN_WITH_INDEX = {
    "id": "convfinqa_indexed_0",
    "pre_text": ["test data."],
    "table": [["A", "B"], ["1", "2"]],
    "post_text": [],
    "qa_0": {
        "question": "What is A?",
        "answer": "1",
    },
    "qa_1": {
        "question": "What is B?",
        "answer": "2",
    },
}


def _write_convfinqa_json(json_path: Path, records: list[dict]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests for turn_index_from_id
# ---------------------------------------------------------------------------

class TestTurnIndexFromId:
    """Tests for convfinqa_adapter.turn_index_from_id()."""

    def test_extracts_trailing_digit(self) -> None:
        assert turn_index_from_id("convfinqa_train_5_0") == 0
        assert turn_index_from_id("convfinqa_train_5_1") == 1
        assert turn_index_from_id("sample_12") == 12

    def test_no_trailing_digit(self) -> None:
        assert turn_index_from_id("convfinqa_train") is None
        assert turn_index_from_id("no_number_here_") is None

    def test_empty_string(self) -> None:
        assert turn_index_from_id("") is None

    def test_only_number(self) -> None:
        assert turn_index_from_id("_42") == 42


# ---------------------------------------------------------------------------
# Tests for active_turn
# ---------------------------------------------------------------------------

class TestActiveTurn:
    """Tests for convfinqa_adapter.active_turn()."""

    def test_single_qa_dict(self) -> None:
        """When sample has a single 'qa' dict, return it directly."""
        turn_key, qa = active_turn(SAMPLE_SINGLE_TURN, SAMPLE_SINGLE_TURN["id"])
        assert turn_key is None
        assert qa["question"] == "What was the revenue increase from Q1 to Q2?"
        assert qa["answer"] == "$20M"

    def test_multi_turn_uses_id_index(self) -> None:
        """When sample id ends with _1, the qa_1 turn should be selected."""
        sample = dict(SAMPLE_MULTI_TURN_WITH_INDEX)
        sample["id"] = "test_1"
        turn_key, qa = active_turn(sample, "test_1")
        assert turn_key == "qa_1"
        assert qa["question"] == "What is B?"

    def test_multi_turn_falls_back_to_last(self) -> None:
        """When the id doesn't hint at a turn, fall back to last qa_N."""
        sample = dict(SAMPLE_MULTI_TURN_WITH_INDEX)
        turn_key, qa = active_turn(sample, "convfinqa_indexed_0")
        # id ends with _0, so qa_0 is selected
        assert turn_key == "qa_0"
        assert qa["question"] == "What is A?"

    def test_no_qa_keys_returns_empty(self) -> None:
        """When there are no qa fields, return an empty dict."""
        sample = {"id": "empty", "pre_text": [], "post_text": [], "table": []}
        turn_key, qa = active_turn(sample, "empty")
        assert turn_key is None
        assert qa == {}

    def test_multi_turn_last_key_fallback(self) -> None:
        """When id has no numeric suffix, use the last qa_N key."""
        sample = {
            "id": "no_suffix",
            "pre_text": [],
            "post_text": [],
            "table": [],
            "qa_0": {"question": "First?", "answer": "A"},
            "qa_1": {"question": "Second?", "answer": "B"},
            "qa_2": {"question": "Third?", "answer": "C"},
        }
        turn_key, qa = active_turn(sample, "no_suffix")
        assert turn_key == "qa_2"
        assert qa["answer"] == "C"


# ---------------------------------------------------------------------------
# Tests for history_to_text
# ---------------------------------------------------------------------------

class TestHistoryToText:
    """Tests for convfinqa_adapter.history_to_text()."""

    def test_no_history_for_first_turn(self) -> None:
        """qa_0 is the first turn so there's no prior history."""
        result = history_to_text(SAMPLE_MULTI_TURN, "qa_0")
        assert result == ""

    def test_history_includes_prior_turns(self) -> None:
        """When current turn is qa_1, history should include qa_0."""
        result = history_to_text(SAMPLE_MULTI_TURN, "qa_1")
        assert "qa_0 question:" in result
        assert "What was the revenue in 2020?" in result
        assert "qa_0 answer:" in result
        assert "$500M" in result
        # Should NOT include qa_1 itself
        assert "qa_1" not in result

    def test_history_for_none_turn_key(self) -> None:
        """When current_turn_key is None, all turns are included."""
        result = history_to_text(SAMPLE_MULTI_TURN, None)
        assert "qa_0 question:" in result
        assert "qa_1 question:" in result

    def test_empty_sample(self) -> None:
        """No qa_N keys means empty history."""
        result = history_to_text({"pre_text": [], "table": []}, "qa_0")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests for turn_number
# ---------------------------------------------------------------------------

class TestTurnNumber:
    """Tests for convfinqa_adapter.turn_number()."""

    def test_valid_turn_keys(self) -> None:
        assert turn_number("qa_0") == 0
        assert turn_number("qa_5") == 5
        assert turn_number("qa_12") == 12

    def test_none_input(self) -> None:
        assert turn_number(None) is None

    def test_empty_string(self) -> None:
        assert turn_number("") is None

    def test_invalid_format(self) -> None:
        assert turn_number("qa") is None
        assert turn_number("question_0") is None


# ---------------------------------------------------------------------------
# Tests for convfinqa_gold_answer_info
# ---------------------------------------------------------------------------

class TestConvfinqaGoldAnswerInfo:
    """Tests for convfinqa_adapter.convfinqa_gold_answer_info()."""

    def test_single_turn_structure(self) -> None:
        qa = SAMPLE_SINGLE_TURN["qa"]
        info = convfinqa_gold_answer_info(SAMPLE_SINGLE_TURN, qa, None, "$20M")
        assert info["primary_answer"] == "$20M"
        assert info["execution_answer"] == "20.0"
        assert isinstance(info["intermediate_answers"], list)
        assert isinstance(info["raw_answer_fields"], dict)

    def test_multi_turn_with_annotation(self) -> None:
        qa = SAMPLE_MULTI_TURN["qa_1"]
        info = convfinqa_gold_answer_info(SAMPLE_MULTI_TURN, qa, "qa_1", "$10M")
        assert info["primary_answer"] == "$10M"
        # annotation.answer_list_1 should be used as intermediate_answers
        assert "$80M" in info["intermediate_answers"]
        assert "$70M" in info["intermediate_answers"]
        assert "$10M" in info["intermediate_answers"]

    def test_raw_answer_fields_populated(self) -> None:
        qa = SAMPLE_MULTI_TURN["qa_1"]
        info = convfinqa_gold_answer_info(SAMPLE_MULTI_TURN, qa, "qa_1", "$10M")
        raw = info["raw_answer_fields"]
        assert raw["answer"] == "$10M"
        assert raw["exe_ans"] == 10.0


# ---------------------------------------------------------------------------
# Tests for build_records
# ---------------------------------------------------------------------------

class TestBuildRecords:
    """Tests for convfinqa_adapter.build_records()."""

    def test_build_records_basic_structure(self, tmp_path: Path) -> None:
        """Each record must conform to the NormalizedSample schema."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN, SAMPLE_MULTI_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert len(records) == 2
        for rec in records:
            assert_normalized_sample(
                rec,
                source=SOURCE,
                task_type=TASK_TYPE,
                task_domain=TASK_DOMAIN,
                language=LANGUAGE,
            )

    def test_build_records_id_preserved(self, tmp_path: Path) -> None:
        """Record ids come from the raw sample."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["id"] == "convfinqa_single_0"

    def test_build_records_question_from_active_turn(self, tmp_path: Path) -> None:
        """Question should come from the active turn's qa."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["question"] == "What was the revenue increase from Q1 to Q2?"

    def test_build_records_answer_resolution(self, tmp_path: Path) -> None:
        """reference_answer should be resolved from qa.answer or qa.exe_ans."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["reference_answer"] == "$20M"

    def test_build_records_context_includes_table_and_text(self, tmp_path: Path) -> None:
        """Context should combine pre_text, table, post_text, and history."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        ctx = records[0]["context"]
        assert "quarterly results" in ctx
        assert "Revenue" in ctx
        assert "$100M" in ctx
        assert "revenue increased" in ctx

    def test_build_records_context_uses_table_ori(self, tmp_path: Path) -> None:
        """When table_ori is present, it should be preferred over table."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_MULTI_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        ctx = records[0]["context"]
        assert "Revenue" in ctx
        assert "$500M" in ctx

    def test_build_records_options_empty(self, tmp_path: Path) -> None:
        """ConvFinQA is a reasoning task, so options should be empty."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["options"] == []

    def test_build_records_metadata_gold_answer_info(self, tmp_path: Path) -> None:
        """Metadata must include gold_answer_info."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert_gold_answer_info(records[0]["metadata"])

    def test_build_records_metadata_raw_split(self, tmp_path: Path) -> None:
        """raw_split should be derived from the filename (e.g., 'train' from train_turn.json)."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["metadata"]["raw_split"] == "train"

    def test_build_records_metadata_current_turn_key(self, tmp_path: Path) -> None:
        """Metadata should store the current_turn_key."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        # Single-turn qa: current_turn_key is None → stored as ""
        assert records[0]["metadata"]["current_turn_key"] == ""

    def test_build_records_metadata_raw_sample(self, tmp_path: Path) -> None:
        """Metadata should include the raw sample dict."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        raw = records[0]["metadata"]["raw_sample"]
        assert raw["id"] == "convfinqa_single_0"

    def test_build_records_multiple_files(self, tmp_path: Path) -> None:
        """Records from multiple *_turn.json files are collected."""
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN],
        )
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "dev_turn.json",
            [SAMPLE_MULTI_TURN],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert len(records) == 2
        raw_splits = {rec["metadata"]["raw_split"] for rec in records}
        assert raw_splits == {"train", "dev"}

    def test_build_records_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty list when no *_turn.json files exist."""
        (tmp_path / "ConvFinQA").mkdir(parents=True)

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records == []

    def test_build_records_fallback_id(self, tmp_path: Path) -> None:
        """When 'id' is missing, a generated id should be used."""
        no_id = {
            "pre_text": [],
            "table": [],
            "post_text": [],
            "qa": {"question": "Test?", "answer": "yes"},
        }
        _write_convfinqa_json(
            tmp_path / "ConvFinQA" / "train_turn.json",
            [no_id],
        )

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path):
            records = build_records()

        assert records[0]["id"] == "convfinqa_train_0"


# ---------------------------------------------------------------------------
# Tests for normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    """Tests for convfinqa_adapter.normalize()."""

    def test_normalize_writes_split_files(self, tmp_path: Path) -> None:
        """normalize() should produce train.jsonl, val.jsonl, test.jsonl."""
        _write_convfinqa_json(
            tmp_path / "raw" / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN, SAMPLE_MULTI_TURN, SAMPLE_MULTI_TURN_WITH_INDEX],
        )
        output_dir = tmp_path / "output"

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path / "raw"):
            report = normalize(output_dir=output_dir, seed=42)

        assert_split_report(report, SOURCE)
        for split_name in ("train", "val", "test"):
            split_path = output_dir / f"{split_name}.jsonl"
            assert split_path.exists(), f"{split_name}.jsonl was not created"

    def test_normalize_output_records_valid(self, tmp_path: Path) -> None:
        """Every record in the output files must conform to the schema."""
        _write_convfinqa_json(
            tmp_path / "raw" / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN, SAMPLE_MULTI_TURN, SAMPLE_MULTI_TURN_WITH_INDEX],
        )
        output_dir = tmp_path / "output"

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path / "raw"):
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
        """The total across splits equals the number of input samples."""
        _write_convfinqa_json(
            tmp_path / "raw" / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN, SAMPLE_MULTI_TURN, SAMPLE_MULTI_TURN_WITH_INDEX],
        )
        output_dir = tmp_path / "output"

        with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path / "raw"):
            report = normalize(output_dir=output_dir, seed=42)

        assert report["total_count"] == 3

    def test_normalize_deterministic(self, tmp_path: Path) -> None:
        """Running normalize twice with the same seed yields identical output."""
        _write_convfinqa_json(
            tmp_path / "raw" / "ConvFinQA" / "train_turn.json",
            [SAMPLE_SINGLE_TURN, SAMPLE_MULTI_TURN, SAMPLE_MULTI_TURN_WITH_INDEX],
        )

        results = []
        for i in range(2):
            output_dir = tmp_path / f"output_{i}"
            with mock.patch("adapters.convfinqa_adapter.RAW_ROOT", tmp_path / "raw"):
                normalize(output_dir=output_dir, seed=42)
            split_records = {}
            for split_name in ("train", "val", "test"):
                with (output_dir / f"{split_name}.jsonl").open("r") as f:
                    split_records[split_name] = [json.loads(line) for line in f]
            results.append(split_records)

        for split_name in ("train", "val", "test"):
            assert results[0][split_name] == results[1][split_name]
