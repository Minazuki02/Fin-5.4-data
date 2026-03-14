from __future__ import annotations

import json
import re
from pathlib import Path

from adapters.common import (
    NORMALIZED_ROOT,
    RAW_ROOT,
    build_gold_answer_info,
    first_non_empty,
    join_sections,
    make_sample,
    read_json,
    table_to_text,
    to_text,
)
from utils.splitter import split_and_write_dataset


SOURCE = "convfinqa"
TASK_TYPE = "reasoning"
TASK_DOMAIN = "financial_reasoning"
LANGUAGE = "en"


def turn_index_from_id(sample_id: str) -> int | None:
    """Extract the active conversation turn index from a ConvFinQA sample id."""
    match = re.search(r"_(\d+)$", sample_id)
    return int(match.group(1)) if match else None


def active_turn(sample: dict, sample_id: str) -> tuple[str | None, dict]:
    """Resolve the current turn payload for a ConvFinQA sample."""
    qa = sample.get("qa")
    if isinstance(qa, dict):
        return None, qa

    turn_index = turn_index_from_id(sample_id)
    if turn_index is not None:
        turn_key = f"qa_{turn_index}"
        turn = sample.get(turn_key)
        if isinstance(turn, dict):
            return turn_key, turn

    turn_keys = sorted(
        (key for key in sample.keys() if re.fullmatch(r"qa_\d+", key)),
        key=lambda name: int(name.split("_")[1]),
    )
    if turn_keys:
        last_key = turn_keys[-1]
        turn = sample.get(last_key)
        if isinstance(turn, dict):
            return last_key, turn
    return None, {}


def history_to_text(sample: dict, current_turn_key: str | None) -> str:
    """Render conversation history that precedes the current turn."""
    lines: list[str] = []
    for key in sorted(
        (name for name in sample.keys() if re.fullmatch(r"qa_\d+", name)),
        key=lambda name: int(name.split("_")[1]),
    ):
        if current_turn_key is not None and key >= current_turn_key:
            break
        turn = sample.get(key)
        if not isinstance(turn, dict):
            continue
        question = to_text(turn.get("question"))
        answer = to_text(turn.get("answer"))
        if question:
            lines.append(f"{key} question: {question}")
        if answer:
            lines.append(f"{key} answer: {answer}")
    return "\n".join(lines)


def turn_number(current_turn_key: str | None) -> int | None:
    if not current_turn_key or not re.fullmatch(r"qa_\d+", current_turn_key):
        return None
    return int(current_turn_key.split("_")[1])


def convfinqa_gold_answer_info(sample: dict, qa: dict, current_turn_key: str | None, primary_answer: str) -> dict:
    annotation = sample.get("annotation", {})
    current_turn_index = turn_number(current_turn_key)
    current_answer_list = None
    if current_turn_index is not None:
        current_answer_list = annotation.get(f"answer_list_{current_turn_index}")

    intermediate_answers = (
        current_answer_list
        or qa.get("answer_list")
        or annotation.get("answer_list")
        or qa.get("exe_ans_list")
        or annotation.get("exe_ans_list")
        or []
    )
    return build_gold_answer_info(
        primary_answer=primary_answer,
        intermediate_answers=intermediate_answers,
        execution_answer=qa.get("exe_ans"),
        raw_answer_fields={
            "answer": qa.get("answer"),
            "answer_list": qa.get("answer_list"),
            "exe_ans": qa.get("exe_ans"),
            "exe_ans_list": qa.get("exe_ans_list"),
            "annotation_answer_list": annotation.get("answer_list"),
            "annotation_exe_ans_list": annotation.get("exe_ans_list"),
            "current_annotation_answer_list": current_answer_list,
            "annotation_answer_list_0": annotation.get("answer_list_0"),
            "annotation_answer_list_1": annotation.get("answer_list_1"),
        },
    )


def build_records() -> list[dict]:
    records: list[dict] = []
    for json_path in sorted((RAW_ROOT / "ConvFinQA").glob("*_turn.json")):
        raw_split = json_path.stem.replace("_turn", "")
        for row_index, sample in enumerate(read_json(json_path)):
            sample_id = sample.get("id") or f"{SOURCE}_{raw_split}_{row_index}"
            current_turn_key, qa = active_turn(sample, str(sample_id))
            answer = first_non_empty(qa.get("answer"), qa.get("exe_ans"))
            records.append(
                make_sample(
                    id=sample_id,
                    source=SOURCE,
                    split="",
                    task_type=TASK_TYPE,
                    task_domain=TASK_DOMAIN,
                    language=LANGUAGE,
                    question=qa.get("question", ""),
                    context=join_sections(
                        "\n".join(sample.get("pre_text", [])),
                        table_to_text(sample.get("table_ori") or sample.get("table")),
                        "\n".join(sample.get("post_text", [])),
                        history_to_text(sample, current_turn_key),
                    ),
                    options=[],
                    reference_answer=answer,
                    metadata={
                        "raw_split": raw_split,
                        "current_turn_key": current_turn_key or "",
                        "gold_answer_info": convfinqa_gold_answer_info(sample, qa, current_turn_key, answer),
                        "raw_sample": sample,
                    },
                )
            )
    return records


def normalize(output_dir: Path | None = None, seed: int = 42) -> dict[str, object]:
    destination = output_dir or (NORMALIZED_ROOT / SOURCE)
    return split_and_write_dataset(source=SOURCE, records=build_records(), output_dir=destination, seed=seed)


def main() -> int:
    report = normalize()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
