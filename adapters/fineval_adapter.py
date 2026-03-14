from __future__ import annotations

import json
from pathlib import Path

from adapters.common import (
    NORMALIZED_ROOT,
    RAW_ROOT,
    build_gold_answer_info,
    iter_csv_rows,
    iter_zip_csv_rows,
    list_zip_members,
    make_sample,
    mcq_options_from_row,
)
from utils.splitter import split_and_write_dataset


SOURCE = "fineval"
TASK_TYPE = "mcq"
TASK_DOMAIN = "financial_knowledge"
LANGUAGE = "zh"


def build_records() -> list[dict]:
    zip_path = RAW_ROOT / "FinEval" / "FinEval.zip"
    extracted_root = RAW_ROOT / "FinEval"
    records: list[dict] = []

    if zip_path.exists():
        members = ((member_name, iter_zip_csv_rows(zip_path, member_name)) for member_name in list_zip_members(zip_path, ".csv"))
    else:
        members = (
            (str(csv_path.relative_to(extracted_root)).replace("\\", "/"), iter_csv_rows(csv_path))
            for csv_path in sorted(extracted_root.rglob("*.csv"))
        )

    for member_name, rows in members:
        raw_split = Path(member_name).parts[0]
        stem = Path(member_name).stem
        for row_index, row in enumerate(rows):
            row_id = row.get("id") or str(row_index)
            answer = row.get("answer", "")
            records.append(
                make_sample(
                    id=f"{SOURCE}_{raw_split}_{stem}_{row_id}",
                    source=SOURCE,
                    split="",
                    task_type=TASK_TYPE,
                    task_domain=TASK_DOMAIN,
                    language=LANGUAGE,
                    question=row.get("question", ""),
                    context="",
                    options=mcq_options_from_row(row),
                    reference_answer=answer,
                    metadata={
                        "raw_split": raw_split,
                        "member_name": member_name,
                        "gold_answer_info": build_gold_answer_info(
                            primary_answer=answer,
                            raw_answer_fields={"answer": row.get("answer", "")},
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
