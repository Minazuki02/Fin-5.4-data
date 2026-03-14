from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = REPO_ROOT / "prompts"
PROMPT_FILES = {
    "mcq": "mcq_prompt.txt",
    "reasoning": "reasoning_prompt.txt",
    "instruction": "instruction_prompt.txt",
    "sentiment": "sentiment_prompt.txt",
}
FINANCEIQ_ACCOUNTING_TAX_SUBJECTS = {
    "注册会计师（CPA）.csv",
    "税务师.csv",
}
FINANCEIQ_REGULATION_SUBJECTS = {
    "银行从业资格.csv",
    "证券从业资格.csv",
    "基金从业资格.csv",
    "保险从业资格CICE.csv",
    "期货从业资格.csv",
}
FINANCEIQ_QUANT_SUBJECTS = {
    "精算师-金融数学.csv",
}
METADATA_KEYS = [
    "table",
    "table_ori",
    "program",
    "program_re",
    "derivation",
    "steps",
    "gold_inds",
    "evidence",
    "history",
    "conversation",
    "paragraphs",
    "pre_text",
    "post_text",
]
MAX_METADATA_CHARS = 4000
MAX_BLOCK_CHARS = 6000


@lru_cache(maxsize=None)
def load_template(file_name: str) -> str:
    """Load one prompt template by file name."""
    return (PROMPT_ROOT / file_name).read_text(encoding="utf-8")


def template_for_sample(sample: dict[str, Any], prompt_version: str) -> str:
    """Resolve the prompt template for one sample without affecting other datasets."""
    task_type = str(sample.get("task_type", "")).strip()
    source = str(sample.get("source", "")).strip().lower()
    if source == "fincuge_instruction" and task_type == "instruction":
        if prompt_version == "fincuge_instruction_hybrid_v1":
            task_name = raw_task_name(sample)
            if task_name == "FINRE":
                return "fincuge_instruction_prompt_finre_simple_v1.txt"
            return "fincuge_instruction_prompt_v2.txt"
        if prompt_version == "fincuge_instruction_finre_simple_v1":
            task_name = raw_task_name(sample)
            if task_name == "FINRE":
                return "fincuge_instruction_prompt_finre_simple_v1.txt"
            return "fincuge_instruction_prompt_v2.txt"
        if prompt_version == "fincuge_instruction_finre_v1":
            task_name = raw_task_name(sample)
            if task_name == "FINRE":
                return "fincuge_instruction_prompt_finre_v1.txt"
            return "fincuge_instruction_prompt_v2.txt"
        if prompt_version == "fincuge_instruction_router_v1":
            task_name = raw_task_name(sample)
            if task_name == "FINNA":
                return "fincuge_instruction_prompt_summary_v1.txt"
            if task_name in {"FINFE", "FINNL", "FINRE", "FINCQA"}:
                return "fincuge_instruction_prompt_label_v1.txt"
            return "fincuge_instruction_prompt_extract_v1.txt"
        if prompt_version == "fincuge_instruction_v2":
            return "fincuge_instruction_prompt_v2.txt"
        return "fincuge_instruction_prompt_v1.txt"
    if source == "finance_alpaca" and task_type == "instruction":
        if prompt_version == "finance_alpaca_v2":
            return "finance_alpaca_instruction_prompt_v2.txt"
        return "finance_alpaca_instruction_prompt_v1.txt"
    if source == "convfinqa" and task_type == "reasoning":
        return "convfinqa_reasoning_prompt.txt"
    if source == "finqa" and task_type == "reasoning":
        return "finqa_reasoning_prompt.txt"
    if source == "tatqa" and task_type == "reasoning":
        return "tatqa_reasoning_prompt.txt"
    if source == "financeiq" and task_type == "mcq":
        subject = subject_name(sample)
        if prompt_version == "financeiq_pilot_v4":
            if subject in FINANCEIQ_ACCOUNTING_TAX_SUBJECTS:
                return "financeiq_mcq_prompt_v4_accounting_tax.txt"
            if subject in FINANCEIQ_REGULATION_SUBJECTS:
                return "financeiq_mcq_prompt_v4_regulation.txt"
            if subject in FINANCEIQ_QUANT_SUBJECTS:
                return "financeiq_mcq_prompt_v4_quant.txt"
            return "financeiq_mcq_prompt_v4_general.txt"
        if prompt_version == "financeiq_pilot_v3":
            return "financeiq_mcq_prompt_v3.txt"
        if prompt_version == "financeiq_pilot_v2":
            return "financeiq_mcq_prompt_v2.txt"
        return "financeiq_mcq_prompt_v1.txt"
    if source == "financial_phrasebank" and task_type == "sentiment":
        if prompt_version == "financial_phrasebank_pilot_v2":
            return "financial_phrasebank_sentiment_prompt_v2.txt"
        if prompt_version == "financial_phrasebank_pilot_v1":
            return "financial_phrasebank_sentiment_prompt_v1.txt"
        if prompt_version == "financial_phrasebank_full_v1":
            return "financial_phrasebank_sentiment_prompt_v1.txt"
        if prompt_version == "financial_phrasebank_full_v2":
            return "financial_phrasebank_sentiment_prompt_v2.txt"
        return "financial_phrasebank_sentiment_prompt_v1.txt"
    return PROMPT_FILES[task_type]


def truncate_text(text: str, limit: int) -> str:
    """Truncate long prompt blocks to keep prompts bounded."""
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n[TRUNCATED]"


def stringify(value: Any) -> str:
    """Convert nested values into a prompt-safe string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def extract_relevant_metadata(sample: dict) -> str:
    """Extract only reasoning-relevant metadata fields for prompting."""
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        return "N/A"

    selected: dict[str, Any] = {}
    raw_sample = metadata.get("raw_sample", {})

    for key in METADATA_KEYS:
        if key in metadata:
            selected[key] = metadata[key]
        elif isinstance(raw_sample, dict) and key in raw_sample:
            selected[key] = raw_sample[key]

    if not selected:
        return "N/A"

    text = json.dumps(selected, ensure_ascii=False, indent=2)
    return truncate_text(text, MAX_METADATA_CHARS)


def subject_name(sample: dict[str, Any]) -> str:
    """Return a compact subject/file hint for datasets that carry it."""
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        direct = str(metadata.get("file_name", "")).strip()
        if direct:
            return direct
        raw_sample = metadata.get("raw_sample", {})
        if isinstance(raw_sample, dict):
            nested = str(raw_sample.get("file_name", "")).strip()
            if nested:
                return nested
    return "N/A"


def raw_task_name(sample: dict[str, Any]) -> str:
    """Return the upstream task identifier if present."""
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        raw_sample = metadata.get("raw_sample", {})
        if isinstance(raw_sample, dict):
            value = str(raw_sample.get("task", "")).strip()
            if value:
                return value
    return "N/A"


def raw_task_desc(sample: dict[str, Any]) -> str:
    """Return the upstream task description if present."""
    metadata = sample.get("metadata", {})
    if isinstance(metadata, dict):
        raw_sample = metadata.get("raw_sample", {})
        if isinstance(raw_sample, dict):
            value = str(raw_sample.get("desc", "")).strip()
            if value:
                return value
    return "N/A"


def build_prompt(sample: dict, prompt_version: str) -> str:
    """Render the task-type prompt for one sample."""
    task_type = sample["task_type"]
    template = load_template(template_for_sample(sample, prompt_version))

    options = sample.get("options") or []
    options_block = "\n".join(str(item) for item in options) if options else "N/A"
    context_block = stringify(sample.get("context", "")) or "N/A"
    metadata_block = extract_relevant_metadata(sample)

    prompt = template.format(
        prompt_version=prompt_version,
        task_type=task_type,
        task_domain=sample.get("task_domain", ""),
        language=sample.get("language", ""),
        subject_name=subject_name(sample),
        raw_task_name=raw_task_name(sample),
        raw_task_desc=raw_task_desc(sample),
        question=stringify(sample.get("question", "")),
        context=truncate_text(context_block, MAX_BLOCK_CHARS),
        context_block=truncate_text(context_block, MAX_BLOCK_CHARS),
        options=truncate_text(options_block, MAX_BLOCK_CHARS),
        options_block=truncate_text(options_block, MAX_BLOCK_CHARS),
        metadata_block=metadata_block,
    )
    return prompt
