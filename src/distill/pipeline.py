from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.distill.cache import is_filled, load_existing_outputs
from src.distill.parser import parse_response
from src.distill.prompt_builder import build_prompt
from src.distill.provider import OpenAICompatibleProvider
from src.utils.io import ensure_parent, iter_jsonl


class DistillPipeline:
    """Run teacher-model distillation over normalized train/val splits."""

    def __init__(self, config: dict[str, Any], provider: OpenAICompatibleProvider, logger: Any) -> None:
        self.config = config
        self.provider = provider
        self.logger = logger
        self.max_concurrency = int(config["api"]["max_concurrency"])
        self.prompt_version = str(config["generation"]["prompt_version"])
        self.skip_if_already_filled = bool(config["processing"]["skip_if_already_filled"])
        self.write_raw_response = bool(config["processing"]["write_raw_response"])
        self.chunk_size = int(config["processing"].get("chunk_size", 64))
        self.progress_log_every = int(config["processing"].get("progress_log_every", 1024))

    def process_all(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        sources: list[str] | None,
        splits: list[str],
        max_samples: int | None,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Process all requested source/split files and return a summary report."""
        source_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
        if sources:
            source_dirs = [path for path in source_dirs if path.name in set(sources)]

        report = {
            "success_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "api_failed_count": 0,
            "latencies_ms": [],
            "source_counts": {},
            "task_type_counts": {},
            "files": [],
        }

        for source_dir in source_dirs:
            for split in splits:
                input_path = source_dir / f"{split}.jsonl"
                if not input_path.exists():
                    continue
                output_path = output_dir / source_dir.name / f"{split}.jsonl"
                file_report = self.process_file(
                    input_path=input_path,
                    output_path=output_path,
                    max_samples=max_samples,
                    overwrite=overwrite,
                )
                report["files"].append(file_report)
                report["success_count"] += file_report["success_count"]
                report["failure_count"] += file_report["failure_count"]
                report["skipped_count"] += file_report["skipped_count"]
                report["api_failed_count"] += file_report["api_failed_count"]
                report["latencies_ms"].extend(file_report["latencies_ms"])

                source_name = source_dir.name
                report["source_counts"][source_name] = report["source_counts"].get(source_name, 0) + file_report["processed_count"]
                for task_type, count in file_report["task_type_counts"].items():
                    report["task_type_counts"][task_type] = report["task_type_counts"].get(task_type, 0) + count

        avg_latency = statistics.mean(report["latencies_ms"]) if report["latencies_ms"] else 0.0
        report["average_latency_ms"] = avg_latency
        report.pop("latencies_ms")
        return report

    def process_file(self, *, input_path: Path, output_path: Path, max_samples: int | None, overwrite: bool) -> dict[str, Any]:
        """Process one normalized split file and write the distilled output file."""
        ensure_parent(output_path)
        existing = load_existing_outputs(output_path)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        file_report = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "processed_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "api_failed_count": 0,
            "latencies_ms": [],
            "task_type_counts": {},
        }
        next_log_at = self.progress_log_every

        with tmp_path.open("w", encoding="utf-8") as out_handle, ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            chunk: list[dict[str, Any]] = []
            for index, record in enumerate(iter_jsonl(input_path)):
                if max_samples is not None and index >= max_samples:
                    if chunk:
                        self._process_chunk(chunk, out_handle, existing, file_report, overwrite, executor)
                        next_log_at = self._maybe_log_progress(file_report, next_log_at)
                        chunk.clear()
                    passthrough = self._passthrough_record(record, existing)
                    out_handle.write(json.dumps(passthrough, ensure_ascii=False) + "\n")
                    continue

                chunk.append(record)
                if len(chunk) >= self.chunk_size:
                    self._process_chunk(chunk, out_handle, existing, file_report, overwrite, executor)
                    next_log_at = self._maybe_log_progress(file_report, next_log_at)
                    chunk.clear()
            if chunk:
                self._process_chunk(chunk, out_handle, existing, file_report, overwrite, executor)
                self._maybe_log_progress(file_report, next_log_at, force=True)

        tmp_path.replace(output_path)
        return file_report

    def _maybe_log_progress(self, file_report: dict[str, Any], next_log_at: int, force: bool = False) -> int:
        """Emit a compact progress log for long-running files."""
        processed = int(file_report["processed_count"])
        if not force and processed < next_log_at:
            return next_log_at
        self.logger.info(
            "progress input=%s processed=%s success=%s failure=%s skipped=%s",
            file_report["input_path"],
            processed,
            file_report["success_count"],
            file_report["failure_count"],
            file_report["skipped_count"],
        )
        if force:
            return next_log_at
        while processed >= next_log_at:
            next_log_at += self.progress_log_every
        return next_log_at

    def _passthrough_record(self, record: dict[str, Any], existing: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Return an untouched record for output continuity when not processing it now."""
        sample_id = str(record.get("id", ""))
        if sample_id and sample_id in existing:
            return existing[sample_id]
        return self._ensure_model_fields(record)

    def _process_chunk(
        self,
        chunk: list[dict[str, Any]],
        out_handle: Any,
        existing: dict[str, dict],
        file_report: dict[str, Any],
        overwrite: bool,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Process one chunk of records while preserving input order."""
        pending: list[dict[str, Any]] = []
        operations: list[tuple[str, dict[str, Any] | str]] = []

        for record in chunk:
            sample_id = str(record.get("id", ""))
            normalized = self._ensure_model_fields(record)
            if not overwrite and self.skip_if_already_filled and sample_id in existing and is_filled(existing[sample_id]):
                operations.append(("ready", existing[sample_id]))
                file_report["skipped_count"] += 1
            elif not overwrite and self.skip_if_already_filled and is_filled(normalized):
                operations.append(("ready", normalized))
                file_report["skipped_count"] += 1
            else:
                operations.append(("pending", sample_id))
                pending.append(normalized)

        results_by_id: dict[str, dict[str, Any]] = {}
        if pending:
            for result in executor.map(self._distill_record, pending):
                results_by_id[result["id"]] = result

        for op_type, payload in operations:
            record = payload if op_type == "ready" else results_by_id[str(payload)]
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            file_report["processed_count"] += 1
            task_type = record.get("task_type", "")
            file_report["task_type_counts"][task_type] = file_report["task_type_counts"].get(task_type, 0) + 1
            distill_meta = record.get("metadata", {}).get("distill", {})
            latency_ms = distill_meta.get("api_latency_ms")
            if isinstance(latency_ms, int) and latency_ms > 0:
                file_report["latencies_ms"].append(latency_ms)
            status = distill_meta.get("parse_status")
            if status == "success":
                file_report["success_count"] += 1
            elif status == "api_failed":
                file_report["failure_count"] += 1
                file_report["api_failed_count"] += 1
            elif status == "failed":
                file_report["failure_count"] += 1

    def _ensure_model_fields(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a record to the expected distilled output schema."""
        normalized = dict(record)
        normalized.setdefault("model_answer", "")
        normalized.setdefault("model_reasoning", "")
        metadata = dict(normalized.get("metadata", {}))
        normalized["metadata"] = metadata
        return normalized

    def _distill_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Call the provider, parse the response, and update one record."""
        prompt = build_prompt(record, prompt_version=self.prompt_version)
        provider_result = self.provider.generate(prompt)

        metadata = dict(record.get("metadata", {}))
        distill_meta = {
            "prompt_version": self.prompt_version,
            "teacher_model": self.provider.model_name,
            "raw_response_text": provider_result.text if self.write_raw_response else "",
            "parse_status": "",
            "parse_error": "",
            "api_success": provider_result.success,
            "api_latency_ms": provider_result.latency_ms,
        }

        updated = dict(record)
        updated["metadata"] = metadata

        if not provider_result.success:
            distill_meta["parse_status"] = "api_failed"
            distill_meta["parse_error"] = provider_result.error
            updated["model_reasoning"] = ""
            updated["model_answer"] = ""
            metadata["distill"] = distill_meta
            return updated

        parsed = parse_response(provider_result.text, updated)
        distill_meta["parse_status"] = parsed.status
        distill_meta["parse_error"] = parsed.error
        updated["model_reasoning"] = parsed.reasoning if parsed.status == "success" else ""
        updated["model_answer"] = parsed.final_answer if parsed.status == "success" else ""
        metadata["distill"] = distill_meta
        return updated
