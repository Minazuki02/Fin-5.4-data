from __future__ import annotations

import csv
import io
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
NORMALIZED_ROOT = REPO_ROOT / "data" / "normalized"


@dataclass(frozen=True)
class NormalizedSample:
    id: str
    source: str
    split: str
    task_type: str
    task_domain: str
    language: str
    question: str
    context: str
    options: list[str]
    reference_answer: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    ensure_dir(output_path.parent)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_csv_rows(path: Path, encoding: str = "utf-8-sig") -> Iterator[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def iter_zip_csv_rows(zip_path: Path, member_name: str, encoding: str = "utf-8-sig") -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding=encoding, newline="")
            reader = csv.DictReader(text_handle)
            for row in reader:
                yield dict(row)


def list_zip_members(zip_path: Path, suffix: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(name for name in archive.namelist() if name.endswith(suffix))


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " | ".join(part for part in (to_text(item) for item in value) if part)
    return json.dumps(value, ensure_ascii=False)


def to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in (to_text(part) for part in value) if item]
    text = to_text(value)
    return [text] if text else []


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return to_text(value)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = to_text(value)
        if text:
            return text
    return ""


def join_sections(*sections: str) -> str:
    return "\n\n".join(section for section in (to_text(item) for item in sections) if section)


def table_to_text(table: Any) -> str:
    if not table:
        return ""
    if isinstance(table, str):
        return table.strip()
    if isinstance(table, dict):
        if "table" in table:
            return table_to_text(table["table"])
        return json.dumps(table, ensure_ascii=False)
    if isinstance(table, list):
        lines: list[str] = []
        for row in table:
            if isinstance(row, list):
                lines.append(" | ".join(to_text(cell) for cell in row))
            else:
                lines.append(to_text(row))
        return "\n".join(line for line in lines if line)
    return to_text(table)


def paragraphs_to_text(paragraphs: Any) -> str:
    if not paragraphs:
        return ""
    if isinstance(paragraphs, str):
        return paragraphs.strip()
    if isinstance(paragraphs, list):
        lines: list[str] = []
        for item in paragraphs:
            if isinstance(item, dict):
                lines.append(to_text(item.get("text")))
            else:
                lines.append(to_text(item))
        return "\n".join(line for line in lines if line)
    return to_text(paragraphs)


def build_gold_answer_info(
    *,
    primary_answer: Any,
    intermediate_answers: Any = None,
    execution_answer: Any = "",
    answer_type: Any = "",
    answer_from: Any = "",
    raw_answer_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "primary_answer": to_text(primary_answer),
        "intermediate_answers": to_text_list(intermediate_answers),
        "execution_answer": to_text(execution_answer),
        "answer_type": to_text(answer_type),
        "answer_from": to_text(answer_from),
        "raw_answer_fields": json_safe(raw_answer_fields or {}),
    }


def mcq_options_from_row(row: dict[str, Any]) -> list[str]:
    option_keys = ["A", "B", "C", "D", "E", "F", "G"]
    options: list[str] = []
    for key in option_keys:
        value = to_text(row.get(key))
        if value:
            options.append(f"{key}. {value}")
    return options


def make_sample(
    *,
    id: str,
    source: str,
    split: Any = "",
    task_type: str,
    task_domain: str,
    language: str,
    question: Any,
    context: Any = "",
    options: list[str] | None = None,
    reference_answer: Any = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = NormalizedSample(
        id=to_text(id),
        source=source,
        split=to_text(split),
        task_type=task_type,
        task_domain=task_domain,
        language=language or "unknown",
        question=to_text(question),
        context=to_text(context),
        options=options or [],
        reference_answer=to_text(reference_answer),
        metadata=metadata or {},
    )
    return sample.to_dict()


def ensure_extracted_zip(url: str, extract_root: Path) -> Path:
    if extract_root.exists():
        return extract_root

    ensure_dir(extract_root.parent)
    zip_path = extract_root.parent / f"{extract_root.name}.zip"
    temp_dir = extract_root.parent / f"{extract_root.name}__tmp"

    if zip_path.exists():
        zip_path.unlink()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    urllib.request.urlretrieve(url, zip_path)
    ensure_dir(temp_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)

    children = [item for item in temp_dir.iterdir() if item.is_dir()]
    if len(children) == 1:
        shutil.move(str(children[0]), str(extract_root))
    else:
        shutil.move(str(temp_dir), str(extract_root))
        temp_dir = None

    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
    if zip_path.exists():
        zip_path.unlink()
    return extract_root
