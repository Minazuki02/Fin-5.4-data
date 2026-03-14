from __future__ import annotations

import json
from pathlib import Path

from adapters.common import (
    NORMALIZED_ROOT,
    RAW_ROOT,
    build_gold_answer_info,
    ensure_extracted_zip,
    first_non_empty,
    join_sections,
    make_sample,
    read_json,
    table_to_text,
)
from utils.splitter import split_and_write_dataset


SOURCE = "finqa"
TASK_TYPE = "reasoning"
TASK_DOMAIN = "financial_reasoning"
LANGUAGE = "en"
ARCHIVE_URL = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"


def dataset_root() -> Path:
    return ensure_extracted_zip(ARCHIVE_URL, RAW_ROOT / "finqa" / "FinQA-main")


def build_records() -> list[dict]:
    root = dataset_root()
    records: list[dict] = []
    split_to_file = {
        "train": root / "dataset" / "train.json",
        "dev": root / "dataset" / "dev.json",
        "test": root / "dataset" / "test.json",
    }

    for raw_split, json_path in split_to_file.items():
        for row_index, sample in enumerate(read_json(json_path)):
            qa = sample.get("qa", {})
            sample_id = sample.get("id") or f"{SOURCE}_{raw_split}_{row_index}"
            answer = first_non_empty(
                qa.get("answer"),
                qa.get("final_result"),
                qa.get("exe_ans"),
                sample.get("answer"),
            )
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
                        table_to_text(sample.get("table")),
                        "\n".join(sample.get("post_text", [])),
                    ),
                    options=[],
                    reference_answer=answer,
                    metadata={
                        "raw_split": raw_split,
                        "gold_answer_info": build_gold_answer_info(
                            primary_answer=answer,
                            execution_answer=qa.get("exe_ans"),
                            raw_answer_fields={
                                "answer": qa.get("answer"),
                                "final_result": qa.get("final_result"),
                                "exe_ans": qa.get("exe_ans"),
                                "sample_answer": sample.get("answer"),
                            },
                        ),
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
