from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import requests
import yaml
from requests import HTTPError

from src.eval.convfinqa_matcher import match_convfinqa_record
from src.evaluation.execution_eval import evaluate_record, gold_answer
from src.evaluation.normalizers import answers_match
from src.utils.io import ensure_parent, iter_jsonl, load_jsonl_by_id, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "finqa_convfinqa_benchmark_prompt.txt"
FINQA_PROMPT_PATH = REPO_ROOT / "prompts" / "finqa_benchmark_prompt.txt"
CONVFINQA_PROMPT_PATH = REPO_ROOT / "prompts" / "convfinqa_benchmark_prompt.txt"
DEFAULT_DATASET_PATHS = {
    "finqa": REPO_ROOT / "data" / "normalized" / "finqa" / "test.jsonl",
    "convfinqa": REPO_ROOT / "data" / "normalized" / "convfinqa" / "test.jsonl",
}
SYSTEM_PROMPT = (
    "You are a careful financial reasoning benchmark model. "
    "Follow the requested output format exactly."
)
NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
UNIT_SUFFIX_RE = re.compile(
    r"(?i)\b(million|billion|thousand|dollars?|usd|shares?|years?|square feet|sq\.?\s*ft\.?)\b"
)


@dataclass(frozen=True)
class ProviderResult:
    success: bool
    text: str
    error: str
    latency_ms: int


@dataclass(frozen=True)
class ExtractedAnswer:
    reasoning: str
    final_answer: str
    method: str


class OpenAIResponsesProvider:
    """Minimal Responses API client built on requests for reproducible benchmarking."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: int,
        retry_times: int,
        max_output_tokens: int,
        reasoning_effort: str,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.retry_times = retry_times
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.session = requests.Session()

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path,
        model_name: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> "OpenAIResponsesProvider":
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        api_conf = config["api"]
        api_base = os.environ.get(api_conf["base_env"], "").strip()
        api_key = os.environ.get(api_conf["key_env"], "").strip()
        if not api_base or not api_key:
            raise ValueError("API_BASE and API_KEY must be set in the environment.")

        timeout = int(api_conf["timeout"])
        retry_times = int(api_conf["retry_times"])
        return cls(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
            retry_times=retry_times,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    def endpoint(self) -> str:
        if self.api_base.endswith("/responses"):
            return self.api_base
        return f"{self.api_base}/responses"

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def _collect_stream_text(self, response: requests.Response) -> str:
        chunks: list[str] = []
        response.encoding = "utf-8"
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace")
            if not line:
                continue

            payload = line[5:].strip() if line.startswith("data:") else line.strip()
            if not payload or payload == "[DONE]":
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            delta = data.get("delta")
            if isinstance(delta, str):
                chunks.append(delta)
                continue

            item = data.get("item")
            if isinstance(item, dict):
                for content in item.get("content", []):
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
        return "".join(chunks).strip()

    def generate(self, prompt: str) -> ProviderResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{SYSTEM_PROMPT}\n\n{prompt}",
                        }
                    ],
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "stream": True,
        }

        last_error = "unknown_error"
        for attempt in range(self.retry_times + 1):
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.endpoint(),
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type.lower():
                    text = self._collect_stream_text(response)
                    return ProviderResult(
                        success=True,
                        text=text,
                        error="",
                        latency_ms=latency_ms,
                    )
                data = response.json()
                return ProviderResult(
                    success=True,
                    text=self._extract_response_text(data),
                    error="",
                    latency_ms=latency_ms,
                )
            except HTTPError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                body = ""
                if exc.response is not None:
                    try:
                        body = exc.response.text[:4000]
                    except Exception:  # noqa: BLE001
                        body = ""
                last_error = f"{exc}; response_body={body}" if body else str(exc)
                if attempt >= self.retry_times:
                    return ProviderResult(False, "", last_error, latency_ms)
                time.sleep(min(2**attempt, 8))
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                if attempt >= self.retry_times:
                    return ProviderResult(False, "", last_error, latency_ms)
                time.sleep(min(2**attempt, 8))

        return ProviderResult(False, "", last_error, 0)


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_convfinqa_prompt_template() -> str:
    return CONVFINQA_PROMPT_PATH.read_text(encoding="utf-8")


def load_finqa_prompt_template() -> str:
    return FINQA_PROMPT_PATH.read_text(encoding="utf-8")


def truncate_text(text: str, limit: int = 24000) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 15] + "\n[TRUNCATED]"


def convfinqa_turn_history(record: dict[str, Any]) -> str:
    meta = record.get("metadata", {})
    raw_sample = meta.get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return "N/A"

    current_turn_key = str(meta.get("current_turn_key", "")).strip()
    if current_turn_key.startswith("qa_"):
        try:
            current_index = int(current_turn_key.split("_")[1])
        except ValueError:
            current_index = None
    else:
        current_index = None

    if current_index is None:
        return "N/A"

    blocks: list[str] = []
    for turn_index in range(current_index):
        turn = raw_sample.get(f"qa_{turn_index}")
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question", "")).strip()
        answer = str(turn.get("answer", "")).strip()
        if question:
            blocks.append(f"Q{turn_index}: {question}")
        if answer:
            blocks.append(f"A{turn_index}: {answer}")
    return "\n".join(blocks) if blocks else "N/A"


def render_prompt(record: dict[str, Any]) -> str:
    dataset_name = str(record.get("source", "")).strip() or "unknown"
    question = truncate_text(record.get("question", ""), 4000) or "N/A"
    context = truncate_text(record.get("context", ""), 24000) or "N/A"
    source = dataset_name.lower()
    if source == "convfinqa":
        template = load_convfinqa_prompt_template()
        return template.format(
            question=question,
            history_block=truncate_text(convfinqa_turn_history(record), 4000) or "N/A",
            context=context,
        )
    if source == "finqa":
        template = load_finqa_prompt_template()
        return template.format(question=question, context=context)
    template = load_prompt_template()
    return template.format(dataset_name=dataset_name, question=question, context=context)


def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_tag(text: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _json_candidate(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_answer(raw_text: str) -> ExtractedAnswer:
    cleaned = _strip_code_fences(raw_text)

    reasoning = _extract_tag(cleaned, "reasoning")
    final_answer = _extract_tag(cleaned, "final_answer")
    if final_answer:
        return ExtractedAnswer(reasoning=reasoning, final_answer=final_answer, method="xml_tags")

    payload = _json_candidate(cleaned)
    if payload:
        reasoning = str(payload.get("reasoning", "")).strip()
        final_answer = str(payload.get("final_answer", "")).strip()
        if final_answer:
            return ExtractedAnswer(reasoning=reasoning, final_answer=final_answer, method="json")

    match = re.search(r"(?is)final answer\s*:\s*(.+)$", cleaned)
    if match:
        final_answer = match.group(1).strip()
        reasoning = cleaned[: match.start()].strip()
        return ExtractedAnswer(reasoning=reasoning, final_answer=final_answer, method="final_answer_marker")

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    final_answer = lines[-1] if lines else ""
    reasoning = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
    return ExtractedAnswer(reasoning=reasoning, final_answer=final_answer, method="last_line_fallback")


def question_expects_percent(question: str) -> bool:
    lowered = str(question or "").lower()
    return any(
        token in lowered
        for token in ("percent", "percentage", "proportion", "portion", "rate", "margin")
    )


def normalize_numeric_surface(text: Any) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = cleaned.replace(",", "").replace("$", "")
    cleaned = UNIT_SUFFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned


def numeric_candidates(text: Any, *, expects_percent: bool) -> tuple[float, ...]:
    cleaned = normalize_numeric_surface(text)
    if not cleaned:
        return ()

    direct = re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", cleaned)
    if direct:
        is_percent = cleaned.endswith("%")
        value = float(cleaned[:-1] if is_percent else cleaned)
        if is_percent:
            return (value, value / 100.0)
        if expects_percent and abs(value) <= 1:
            return (value, value * 100.0)
        return (value,)

    matches = NUMERIC_TOKEN_RE.findall(cleaned)
    if not matches:
        return ()

    token = matches[-1]
    token = token.replace(",", "").replace("$", "")
    is_percent = token.endswith("%")
    value = float(token[:-1] if is_percent else token)
    if is_percent:
        return (value, value / 100.0)
    if expects_percent and abs(value) <= 1:
        return (value, value * 100.0)
    return (value,)


def benchmark_match_finqa(prediction: Any, gold: Any, *, question: str) -> tuple[bool, str, str]:
    expects_percent = question_expects_percent(question)
    pred_candidates = numeric_candidates(prediction, expects_percent=expects_percent)
    gold_candidates = numeric_candidates(gold, expects_percent=expects_percent)

    pred_text = normalize_numeric_surface(prediction)
    gold_text = normalize_numeric_surface(gold)

    if pred_candidates and gold_candidates:
        abs_tol = 0.51 if expects_percent and "." not in gold_text else 0.051 if expects_percent else 1e-4
        rel_tol = 0.01 if expects_percent else 1e-4
        for left in pred_candidates:
            for right in gold_candidates:
                if math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol):
                    normalized_prediction = f"{left:g}%".replace("%", "") if False else pred_text
                    return True, pred_text, gold_text

    if answers_match(pred_text, gold_text):
        return True, pred_text, gold_text
    return False, pred_text, gold_text


def benchmark_execution_match_finqa(record: dict[str, Any], prediction: Any) -> bool:
    execution_gold = gold_execution_answer(record)
    question = str(record.get("question", ""))
    if execution_gold not in ("", None):
        pred_candidates = context_unit_scale_candidates(record, prediction)
        gold_candidates = context_unit_scale_candidates(record, execution_gold)
        for left in pred_candidates:
            for right in gold_candidates:
                if math.isclose(left, right, abs_tol=0.51, rel_tol=0.01):
                    return True
        return False

    matched, _, _ = benchmark_match_finqa(prediction, gold_answer(record), question=question)
    return matched


def gold_execution_answer(record: dict[str, Any]) -> Any:
    meta = record.get("metadata", {})
    gold_info = meta.get("gold_answer_info", {})
    execution = gold_info.get("execution_answer")
    if execution not in ("", None):
        return execution
    raw_sample = meta.get("raw_sample", {})
    if not isinstance(raw_sample, dict):
        return None
    current_turn_key = str(meta.get("current_turn_key", "")).strip()
    if current_turn_key and isinstance(raw_sample.get(current_turn_key), dict):
        return raw_sample[current_turn_key].get("exe_ans")
    qa = raw_sample.get("qa")
    if isinstance(qa, dict):
        return qa.get("exe_ans")
    return None


def context_unit_scale_candidates(record: dict[str, Any], answer: Any) -> tuple[float, ...]:
    question = str(record.get("question", ""))
    if question_expects_percent(question):
        return numeric_candidates(answer, expects_percent=True)

    base = set(numeric_candidates(answer, expects_percent=False))
    context = str(record.get("context", "")).lower()
    if not base:
        return ()
    if "in millions" in context or "amount ( in millions )" in context or "($ in millions" in context:
        base.update(value * 1_000_000 for value in list(base))
    if "in billions" in context or "amount ( in billions )" in context or "($ in billions" in context:
        base.update(value * 1_000_000_000 for value in list(base))
    return tuple(base)


def benchmark_match_convfinqa(record: dict[str, Any], extracted: ExtractedAnswer) -> dict[str, Any]:
    match = match_convfinqa_record(_base_prediction_record(record, extracted.final_answer, extracted.reasoning))

    accuracy_ok = match.strict_match or (
        match.lenient_match and match.mismatch_type == "rounding_or_format"
    )

    reference = gold_answer(record)
    ref_candidates = context_unit_scale_candidates(record, reference)
    pred_candidates = context_unit_scale_candidates(record, extracted.final_answer)
    if not accuracy_ok and ref_candidates and pred_candidates:
        for left in pred_candidates:
            for right in ref_candidates:
                if math.isclose(left, right, abs_tol=0.51, rel_tol=0.01):
                    accuracy_ok = True
                    break
            if accuracy_ok:
                break

    execution_gold = gold_execution_answer(record)
    execution_correct = accuracy_ok
    if execution_gold not in ("", None):
        exec_candidates = context_unit_scale_candidates(record, extracted.final_answer)
        gold_exec_candidates = context_unit_scale_candidates(record, execution_gold)
        execution_correct = False
        for left in exec_candidates:
            for right in gold_exec_candidates:
                if math.isclose(left, right, abs_tol=0.51, rel_tol=0.01):
                    execution_correct = True
                    break
            if execution_correct:
                break

    mismatch_type = "match" if accuracy_ok else match.mismatch_type
    return {
        "correct": accuracy_ok,
        "execution_correct": execution_correct,
        "lenient_match": match.lenient_match,
        "normalized_prediction": match.normalized_prediction,
        "normalized_gold": match.normalized_gold,
        "mismatch_type": mismatch_type,
        "executable": False,
    }


def _base_prediction_record(record: dict[str, Any], final_answer: str, reasoning: str) -> dict[str, Any]:
    cloned = dict(record)
    cloned["model_answer"] = final_answer
    cloned["model_reasoning"] = reasoning
    return cloned


def evaluate_prediction(record: dict[str, Any], extracted: ExtractedAnswer) -> dict[str, Any]:
    source = str(record.get("source", "")).lower()
    reference = gold_answer(record)

    if source == "convfinqa":
        return benchmark_match_convfinqa(record, extracted)

    accuracy_ok, normalized_prediction, normalized_gold = benchmark_match_finqa(
        extracted.final_answer,
        reference,
        question=str(record.get("question", "")),
    )
    execution = evaluate_record(_base_prediction_record(record, extracted.final_answer, extracted.reasoning))
    execution_correct = benchmark_execution_match_finqa(record, extracted.final_answer)
    return {
        "correct": accuracy_ok,
        "execution_correct": execution_correct,
        "lenient_match": execution_correct,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "mismatch_type": "match" if accuracy_ok else "mismatch",
        "executable": bool(execution["executable"]),
    }


def benchmark_output_dir(root: Path, model_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return root / f"{safe_model}_{stamp}"


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total_samples": 0,
            "api_success_rate": 0.0,
            "accuracy": 0.0,
            "execution_accuracy": 0.0,
            "executable_rate": 0.0,
        }

    api_success = sum(1 for record in records if record["api_success"])
    correct = sum(1 for record in records if record["correct"])
    execution_correct = sum(1 for record in records if record["execution_correct"])
    executable = sum(1 for record in records if record["executable"])
    return {
        "total_samples": total,
        "api_success_rate": api_success / total,
        "accuracy": correct / total,
        "execution_accuracy": execution_correct / total,
        "executable_rate": executable / total,
        "correct_count": correct,
        "execution_correct_count": execution_correct,
        "api_success_count": api_success,
        "executable_count": executable,
    }


def evaluate_dataset(
    *,
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    provider: OpenAIResponsesProvider,
    max_samples: int | None,
    max_concurrency: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_records = list(iter_jsonl(input_path))
    if max_samples is not None:
        source_records = source_records[:max_samples]

    existing = {} if overwrite else load_jsonl_by_id(output_path)
    pending = [record for record in source_records if str(record.get("id", "")) not in existing]
    completed = dict(existing)

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt(record)
        provider_result = provider.generate(prompt)
        extracted = extract_answer(provider_result.text) if provider_result.success else ExtractedAnswer("", "", "none")
        evaluation = evaluate_prediction(record, extracted) if provider_result.success else {
            "correct": False,
            "execution_correct": False,
            "lenient_match": False,
            "normalized_prediction": "",
            "normalized_gold": gold_answer(record),
            "mismatch_type": "api_error",
            "executable": False,
        }
        return {
            "id": record.get("id", ""),
            "source": record.get("source", dataset_name),
            "split": record.get("split", "test"),
            "question": record.get("question", ""),
            "context": record.get("context", ""),
            "reference_answer": gold_answer(record),
            "prompt": prompt,
            "api_success": provider_result.success,
            "api_error": provider_result.error,
            "api_latency_ms": provider_result.latency_ms,
            "raw_model_output": provider_result.text,
            "extracted_reasoning": extracted.reasoning,
            "extracted_final_answer": extracted.final_answer,
            "extraction_method": extracted.method,
            "normalized_prediction": evaluation["normalized_prediction"],
            "normalized_gold": evaluation["normalized_gold"],
            "correct": evaluation["correct"],
            "execution_correct": evaluation["execution_correct"],
            "lenient_match": evaluation["lenient_match"],
            "mismatch_type": evaluation["mismatch_type"],
            "executable": evaluation["executable"],
        }

    if pending:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_id = {
                executor.submit(run_one, record): str(record.get("id", ""))
                for record in pending
            }
            for future in as_completed(future_to_id):
                result = future.result()
                completed[result["id"]] = result

    ordered = [completed[str(record.get("id", ""))] for record in source_records if str(record.get("id", "")) in completed]
    write_jsonl(output_path, iter(ordered))
    return {
        "dataset_name": dataset_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        **summarize_predictions(ordered),
    }


def run_benchmark(
    *,
    dataset_paths: dict[str, Path],
    output_dir: Path,
    provider: OpenAIResponsesProvider,
    max_samples: int | None,
    max_concurrency: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_parent(output_dir / "summary.json")

    per_dataset: dict[str, Any] = {}
    dataset_outputs: list[Path] = []
    all_records: list[dict[str, Any]] = []

    for dataset_name, input_path in dataset_paths.items():
        output_path = output_dir / f"{dataset_name}.predictions.jsonl"
        dataset_outputs.append(output_path)
        dataset_summary = evaluate_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
            output_path=output_path,
            provider=provider,
            max_samples=max_samples,
            max_concurrency=max_concurrency,
            overwrite=overwrite,
        )
        per_dataset[dataset_name] = dataset_summary
        all_records.extend(list(iter_jsonl(output_path)))

    summary = {
        "model_name": provider.model_name,
        "prompt_paths": {
            "default": str(PROMPT_PATH),
            "finqa": str(FINQA_PROMPT_PATH),
            "convfinqa": str(CONVFINQA_PROMPT_PATH),
        },
        "datasets": list(dataset_paths.keys()),
        "per_dataset": per_dataset,
        "overall": summarize_predictions(all_records),
        "output_dir": str(output_dir),
        "prediction_files": [str(path) for path in dataset_outputs],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
