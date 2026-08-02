from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from http import client
from typing import Any

from ..config import LLMConfig


class _NonRetryableLLMError(RuntimeError):
    """Raised for HTTP 4xx (except 429) and response bodies that cannot yield a completion."""


def _format_api_error(data: dict[str, Any]) -> str | None:
    err = data.get("error")
    if err is None:
        return None
    if isinstance(err, dict):
        return str(err.get("message", err.get("type", json.dumps(err))))
    return str(err)


def _retryable_429_detail(detail: str) -> bool:
    """429 can mean rate limits (retry) or a dead/invalid key or account (do not retry)."""
    d = detail.lower()
    if "deactivated" in d:
        return False
    if "invalid_api_key" in d or "invalid api key" in d:
        return False
    if "incorrect api key" in d:
        return False
    if "api key" in d and ("invalid" in d or "expired" in d or "not found" in d):
        return False
    if "authentication" in d and ("failed" in d or "invalid" in d):
        return False
    if "revoked" in d or "has been deleted" in d:
        return False
    if "insufficient_quota" in d:
        return False
    return True


def _completion_text_from_payload(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        raise _NonRetryableLLMError(f"LLM returned non-object JSON: {type(data).__name__}")

    api_err = _format_api_error(data)
    if api_err:
        raise _NonRetryableLLMError(f"LLM API error: {api_err}")

    choices = data.get("choices")
    if choices is None:
        preview = json.dumps(data, ensure_ascii=False)[:900]
        raise _NonRetryableLLMError(
            f"LLM response missing 'choices' (keys: {sorted(data.keys())}). Body preview: {preview}"
        )
    if len(choices) == 0:
        raise _NonRetryableLLMError("LLM response has an empty 'choices' list.")

    first = choices[0]
    if not isinstance(first, dict):
        raise _NonRetryableLLMError(f"Unexpected choice entry type: {type(first).__name__}")

    msg = first.get("message")
    if not isinstance(msg, dict):
        raise _NonRetryableLLMError(
            f"Unexpected choice shape (missing message dict): {json.dumps(first)[:500]}"
        )

    content = msg.get("content")
    if content is None:
        raise _NonRetryableLLMError(
            f"LLM message missing 'content': {json.dumps(msg)[:500]}"
        )
    return str(content)


def _load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


@dataclass(slots=True)
class LLMClient:
    config: LLMConfig

    def __post_init__(self) -> None:
        _load_env_file(self.config.env_file)

    def query(self, prompt: str, model_name: str | None = None) -> str:
        api_key = os.environ.get(self.config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key in env var {self.config.api_key_env}. "
                "Populate .env or export the variable before running the pipeline."
            )

        payload = json.dumps(
            {
                "model": model_name or self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            connection = client.HTTPSConnection(self.config.host, timeout=self.config.timeout_seconds)
            headers = {
                "Accept": "application/json",
                "Authorization": api_key.strip().strip('"').strip("'"),
                "User-Agent": random.choice(self.config.user_agents),
                "Content-Type": "application/json",
            }
            try:
                connection.request("POST", self.config.path, payload, headers)
                response = connection.getresponse()
                raw_bytes = response.read()
                status = response.status
                try:
                    data = json.loads(raw_bytes.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    preview = raw_bytes[:500]
                    raise _NonRetryableLLMError(
                        f"LLM returned non-JSON (HTTP {status}): {preview!r}"
                    ) from exc

                if status == 429 or status >= 500:
                    detail = _format_api_error(data) if isinstance(data, dict) else str(data)
                    if not detail:
                        detail = raw_bytes.decode("utf-8", errors="replace")[:800]
                    if status == 429 and not _retryable_429_detail(detail):
                        raise _NonRetryableLLMError(
                            f"LLM HTTP {status}: {detail}\n"
                            f"Update {self.config.api_key_env} in .env (or your shell), or change "
                            "`llm.host` / `llm.model` in the JSON config if you use a different provider."
                        )
                    raise OSError(f"LLM HTTP {status}: {detail}")

                if status >= 400:
                    detail = _format_api_error(data) if isinstance(data, dict) else None
                    if not detail:
                        detail = raw_bytes.decode("utf-8", errors="replace")[:800]
                    raise _NonRetryableLLMError(f"LLM HTTP {status}: {detail}")

                return _completion_text_from_payload(data)
            except _NonRetryableLLMError:
                raise
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.sleep_seconds * (attempt + 1))
            finally:
                try:
                    connection.close()
                except Exception:
                    pass
        raise RuntimeError(f"LLM request failed after {self.config.max_retries} attempts: {last_error}")
