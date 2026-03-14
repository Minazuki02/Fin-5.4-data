from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests import HTTPError


@dataclass(frozen=True)
class ProviderResult:
    """Container for a single provider response."""

    success: bool
    text: str
    error: str
    latency_ms: int


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat completions endpoint with retry logic."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_name: str,
        timeout: int,
        retry_times: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.retry_times = retry_times
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = requests.Session()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleProvider":
        """Build a provider instance from the distill config."""
        api_conf = config["api"]
        gen_conf = config["generation"]

        api_base = os.environ.get(api_conf["base_env"], "").strip()
        api_key = os.environ.get(api_conf["key_env"], "").strip()
        model_name = os.environ.get(api_conf["model_env"], "").strip()

        if not api_base or not api_key or not model_name:
            raise ValueError("API_BASE, API_KEY, and MODEL_NAME must be set in the environment")

        return cls(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            timeout=int(api_conf["timeout"]),
            retry_times=int(api_conf["retry_times"]),
            temperature=float(gen_conf["temperature"]),
            max_tokens=int(gen_conf["max_tokens"]),
        )

    def endpoint(self) -> str:
        """Return the full chat completions endpoint."""
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return f"{self.api_base}/chat/completions"

    def _extract_response_text(self, data: dict[str, Any]) -> str:
        """Extract assistant text from a non-streaming response payload."""
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

    def _collect_stream_text(self, response: requests.Response) -> str:
        """Collect assistant text from a streaming SSE response."""
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
            if line.startswith("data:"):
                payload = line[5:].strip()
            else:
                payload = line.strip()

            if not payload or payload == "[DONE]":
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))

            message = choices[0].get("message")
            if isinstance(message, dict):
                full_content = message.get("content", "")
                if isinstance(full_content, str) and full_content:
                    chunks.append(full_content)

        return "".join(chunks)

    def generate(self, prompt: str) -> ProviderResult:
        """Generate a response from the teacher model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a careful financial reasoning teacher."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
                else:
                    data = response.json()
                    text = self._extract_response_text(data)
                return ProviderResult(success=True, text=text, error="", latency_ms=latency_ms)
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
                    return ProviderResult(success=False, text="", error=last_error, latency_ms=latency_ms)
                time.sleep(min(2**attempt, 8))
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                if attempt >= self.retry_times:
                    return ProviderResult(success=False, text="", error=last_error, latency_ms=latency_ms)
                time.sleep(min(2**attempt, 8))

        return ProviderResult(success=False, text="", error=last_error, latency_ms=0)
