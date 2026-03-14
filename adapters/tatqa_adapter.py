from __future__ import annotations

import json
from pathlib import Path

from adapters.common import NORMALIZED_ROOT, RAW_ROOT, build_gold_answer_info, join_sections, make_sample, read_json, table_to_text
from utils.splitter import split_and_write_dataset


SOURCE = "tatqa"
TASK_TYPE = "reasoning"
TASK_DOMAIN = "financial_reasoning"
LANGUAGE = "en"


def build_records() -> list[dict]:
    records: list[dict] = []
    file_map = {
        "train": RAW_ROOT / "TAT-QA" / "tatqa_dataset_train.json",
        "dev": RAW_ROOT / "TAT-QA" / "tatqa_dataset_dev.json",
        "test_gold": RAW_ROOT / "TAT-QA" / "tatqa_dataset_test_gold.json",
    }

    for raw_split, json_path in file_map.items():
        for sample_index, sample in enumerate(read_json(json_path)):
            context = join_sections(
                "\n".join(paragraph.get("text", "") for paragraph in sample.get("paragraphs", [])),
                table_to_text(sample.get("table")),
            )
            for question_index, question_item in enumerate(sample.get("questions", [])):
                sample_id = question_item.get("uid") or f"{SOURCE}_{raw_split}_{sample_index}_{question_index}"
                answer = question_item.get("answer", "")
                records.append(
                    make_sample(
                        id=sample_id,
                        source=SOURCE,
                        split="",
                        task_type=TASK_TYPE,
                        task_domain=TASK_DOMAIN,
                        language=LANGUAGE,
                        question=question_item.get("question", ""),
                        context=context,
                        options=[],
                        reference_answer=answer,
                        metadata={
                            "raw_split": raw_split,
                            "table_uid": sample.get("table", {}).get("uid", ""),
                            "gold_answer_info": build_gold_answer_info(
                                primary_answer=answer,
                                answer_type=question_item.get("answer_type"),
                                answer_from=question_item.get("answer_from"),
                                raw_answer_fields={
                                    "answer": question_item.get("answer"),
                                    "answer_type": question_item.get("answer_type"),
                                    "answer_from": question_item.get("answer_from"),
                                },
                            ),
                            "raw_sample": sample,
                            "raw_question": question_item,
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
