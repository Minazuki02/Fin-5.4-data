from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from adapters.common import NORMALIZED_ROOT, RAW_ROOT, build_gold_answer_info, make_sample
from utils.splitter import split_and_write_dataset


SOURCE = "fincuge_instruction"
TASK_TYPE = "instruction"
TASK_DOMAIN = "financial_instruction"
LANGUAGE = "zh"


def build_records() -> list[dict]:
    records: list[dict] = []
    data_dir = RAW_ROOT / "FinCUGE-Instruction" / "data"

    for parquet_path in sorted(data_dir.glob("*.parquet")):
        raw_split = parquet_path.stem.split("-")[0]
        dataframe = pd.read_parquet(parquet_path)
        for row_index, row in dataframe.iterrows():
            raw_sample = row.to_dict()
            answer = raw_sample.get("output", "")
            records.append(
                make_sample(
                    id=f"{SOURCE}_{raw_split}_{row_index}",
                    source=SOURCE,
                    split="",
                    task_type=TASK_TYPE,
                    task_domain=TASK_DOMAIN,
                    language=LANGUAGE,
                    question=raw_sample.get("instruction", ""),
                    context=raw_sample.get("input", ""),
                    options=[],
                    reference_answer=answer,
                    metadata={
                        "raw_split": raw_split,
                        "file_name": parquet_path.name,
                        "gold_answer_info": build_gold_answer_info(
                            primary_answer=answer,
                            raw_answer_fields={"output": raw_sample.get("output", "")},
                        ),
                        "raw_sample": raw_sample,
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
