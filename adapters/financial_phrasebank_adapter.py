from __future__ import annotations

import json
import zipfile
from pathlib import Path

from adapters.common import NORMALIZED_ROOT, RAW_ROOT, build_gold_answer_info, make_sample
from utils.splitter import split_and_write_dataset


SOURCE = "financial_phrasebank"
TASK_TYPE = "sentiment"
TASK_DOMAIN = "market_sentiment"
LANGUAGE = "en"
OPTIONS = ["negative", "neutral", "positive"]
QUESTION = "判断下面金融文本的情绪"


def build_records() -> list[dict]:
    zip_path = RAW_ROOT / "financial_phrasebank" / "data" / "FinancialPhraseBank-v1.0.zip"
    records: list[dict] = []

    with zipfile.ZipFile(zip_path) as archive:
        member_names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".txt") and "Sentences_" in name and "__MACOSX" not in name
        )
        for member_name in member_names:
            agreement_split = Path(member_name).stem.replace("Sentences_", "").replace("Agree", "").lower()
            with archive.open(member_name) as raw_handle:
                for row_index, line in enumerate(raw_handle.read().decode("iso-8859-1").splitlines()):
                    if "@" not in line:
                        continue
                    sentence, label = line.rsplit("@", 1)
                    answer = label.strip()
                    records.append(
                        make_sample(
                            id=f"{SOURCE}_{agreement_split}_{row_index}",
                            source=SOURCE,
                            split="",
                            task_type=TASK_TYPE,
                            task_domain=TASK_DOMAIN,
                            language=LANGUAGE,
                            question=QUESTION,
                            context=sentence,
                            options=OPTIONS,
                            reference_answer=answer,
                            metadata={
                                "raw_split": "",
                                "agreement_split": agreement_split,
                                "member_name": member_name,
                                "gold_answer_info": build_gold_answer_info(
                                    primary_answer=answer,
                                    raw_answer_fields={"label": answer},
                                ),
                                "raw_sample": {
                                    "sentence": sentence,
                                    "label": answer,
                                },
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
