from __future__ import annotations

import json
from pathlib import Path

from adapters.common import NORMALIZED_ROOT, RAW_ROOT, build_gold_answer_info, iter_csv_rows, make_sample, mcq_options_from_row
from utils.splitter import split_and_write_dataset


SOURCE = "financeiq"
TASK_TYPE = "mcq"
TASK_DOMAIN = "financial_knowledge"
LANGUAGE = "zh"


def build_records() -> list[dict]:
    raw_root = RAW_ROOT / "FinanceIQ" / "data"
    records: list[dict] = []

    for csv_path in sorted(raw_root.rglob("*.csv")):
        raw_split = csv_path.parent.name
        for row_index, row in enumerate(iter_csv_rows(csv_path)):
            row_id = row.get("") or row.get("id") or str(row_index)
            answer = row.get("Answer", "")
            records.append(
                make_sample(
                    id=f"{SOURCE}_{raw_split}_{csv_path.stem}_{row_id}",
                    source=SOURCE,
                    split="",
                    task_type=TASK_TYPE,
                    task_domain=TASK_DOMAIN,
                    language=LANGUAGE,
                    question=row.get("Question", ""),
                    context="",
                    options=mcq_options_from_row(row),
                    reference_answer=answer,
                    metadata={
                        "raw_split": raw_split,
                        "file_name": csv_path.name,
                        "gold_answer_info": build_gold_answer_info(
                            primary_answer=answer,
                            raw_answer_fields={"Answer": row.get("Answer", "")},
                        ),
                        "raw_sample": row,
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
