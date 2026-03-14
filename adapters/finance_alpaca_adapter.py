from __future__ import annotations

import json
from pathlib import Path

from adapters.common import NORMALIZED_ROOT, RAW_ROOT, build_gold_answer_info, make_sample, read_json
from utils.splitter import split_and_write_dataset


SOURCE = "finance_alpaca"
TASK_TYPE = "instruction"
TASK_DOMAIN = "financial_instruction"
LANGUAGE = "en"


def build_records() -> list[dict]:
    json_path = RAW_ROOT / "finance-alpaca" / "Cleaned_date.json"
    records: list[dict] = []
    for row_index, sample in enumerate(read_json(json_path)):
        answer = sample.get("output", "")
        records.append(
            make_sample(
                id=f"{SOURCE}_train_{row_index}",
                source=SOURCE,
                split="",
                task_type=TASK_TYPE,
                task_domain=TASK_DOMAIN,
                language=LANGUAGE,
                question=sample.get("instruction", ""),
                context=sample.get("input", ""),
                options=[],
                reference_answer=answer,
                metadata={
                    "raw_split": "train",
                    "gold_answer_info": build_gold_answer_info(
                        primary_answer=answer,
                        raw_answer_fields={"output": sample.get("output", "")},
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
