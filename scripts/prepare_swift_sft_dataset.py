from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_parent, iter_jsonl, write_jsonl  # noqa: E402


DEFAULT_INPUT_DIR = REPO_ROOT / "train_data" / "final_mixture_exact"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "train_data" / "swift_final_mixture_exact"

SYSTEM_PROMPTS = {
    "reasoning": "You are a careful financial reasoning assistant. Solve the task using the provided evidence and follow the requested output format exactly.",
    "mcq": "You are a careful financial knowledge assistant. Read the question and options, reason briefly, and output the final choice in the required format.",
    "instruction": "You are a careful financial instruction-following assistant. Follow the task faithfully and use the requested output format exactly.",
    "sentiment": "You are a careful financial sentiment assistant. Classify the text using the provided label set and follow the requested output format exactly.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mixed JSONL training data into ms-swift SFT messages format.")
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    return parser.parse_args()


def format_options(options: Any) -> str:
    if not isinstance(options, list) or not options:
        return ""
    lines: list[str] = []
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        option_text = str(option).strip()
        option_text = option_text.removeprefix(f"{label}.").strip()
        option_text = option_text.removeprefix(f"{label}．").strip()
        option_text = option_text.removeprefix(f"{label}、").strip()
        option_text = option_text.removeprefix(f"{label})").strip()
        lines.append(f"{label}. {option_text}")
    return "\n".join(lines)


def convfinqa_history(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    raw_sample = metadata.get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return ""

    current_turn_key = str(metadata.get("current_turn_key", "")).strip()
    if not current_turn_key.startswith("qa_"):
        return ""

    try:
        current_turn_index = int(current_turn_key.split("_")[1])
    except ValueError:
        return ""

    lines: list[str] = []
    for turn_index in range(current_turn_index):
        turn = raw_sample.get(f"qa_{turn_index}")
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question", "")).strip()
        answer = str(turn.get("answer", "")).strip()
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def build_user_prompt(record: dict[str, Any]) -> str:
    source = str(record.get("source", "")).lower()
    task_type = str(record.get("task_type", "")).lower()
    question = str(record.get("question", "")).strip()
    context = str(record.get("context", "")).strip()
    options = format_options(record.get("options"))

    blocks: list[str] = []

    if task_type == "mcq":
        blocks.append("Task: Choose the single best option.")
        blocks.append(f"Question:\n{question}")
        if options:
            blocks.append(f"Options:\n{options}")
        if context:
            blocks.append(f"Context:\n{context}")
        blocks.append(
            "Return exactly this format:\n<reasoning>\n...\n</reasoning>\n<final_answer>\nA single option letter\n</final_answer>"
        )
        return "\n\n".join(blocks)

    if task_type == "sentiment":
        blocks.append("Task: Classify the sentiment of the financial text.")
        if question:
            blocks.append(f"Instruction:\n{question}")
        if context:
            blocks.append(f"Text:\n{context}")
        if options:
            blocks.append(f"Allowed labels:\n{options}")
        blocks.append(
            "Return exactly this format:\n<reasoning>\n...\n</reasoning>\n<final_answer>\nOne label only\n</final_answer>"
        )
        return "\n\n".join(blocks)

    if source == "convfinqa":
        history = convfinqa_history(record)
        blocks.append("Task: Solve the current ConvFinQA turn.")
        blocks.append(f"Current question:\n{question}")
        if history:
            blocks.append(f"Prior dialogue:\n{history}")
        if context:
            blocks.append(f"Context:\n{context}")
        blocks.append(
            "Return exactly this format:\n<reasoning>\n...\n</reasoning>\n<final_answer>\nThe final answer only\n</final_answer>"
        )
        return "\n\n".join(blocks)

    if task_type == "reasoning":
        blocks.append("Task: Solve the financial reasoning problem.")
        blocks.append(f"Question:\n{question}")
        if context:
            blocks.append(f"Context:\n{context}")
        blocks.append(
            "Return exactly this format:\n<reasoning>\n...\n</reasoning>\n<final_answer>\nThe final answer only\n</final_answer>"
        )
        return "\n\n".join(blocks)

    blocks.append("Task: Follow the instruction.")
    blocks.append(f"Instruction:\n{question}")
    if context:
        blocks.append(f"Additional context:\n{context}")
    if options:
        blocks.append(f"Options:\n{options}")
    blocks.append(
        "Return exactly this format:\n<reasoning>\n...\n</reasoning>\n<final_answer>\nThe final answer only\n</final_answer>"
    )
    return "\n\n".join(blocks)


def build_assistant_response(record: dict[str, Any]) -> str:
    reasoning = str(record.get("model_reasoning", "")).strip()
    final_answer = str(record.get("model_answer", "")).strip()
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<final_answer>\n{final_answer}\n</final_answer>"


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    task_type = str(record.get("task_type", "")).lower()
    system_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["instruction"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(record)},
        {"role": "assistant", "content": build_assistant_response(record)},
    ]
    return {
        "id": record.get("id", ""),
        "messages": messages,
        "source": record.get("source", ""),
        "task_type": record.get("task_type", ""),
        "task_domain": record.get("task_domain", ""),
        "language": record.get("language", ""),
    }


def convert_split(input_path: Path, output_path: Path) -> dict[str, Any]:
    records = list(iter_jsonl(input_path))
    converted = (convert_record(record) for record in records)
    count = write_jsonl(output_path, converted)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "count": count,
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    summary: dict[str, Any] = {"input_dir": str(input_dir), "output_dir": str(output_dir), "splits": {}}
    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")
        output_path = output_dir / f"{split}.jsonl"
        ensure_parent(output_path)
        summary["splits"][split] = convert_split(input_path, output_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
