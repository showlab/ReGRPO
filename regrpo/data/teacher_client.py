# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Teacher VLM client for Reflection-of-Thought (RoT) data generation.

The Structured Reflective Data Engine queries a teacher vision-language model
(GPT-4o by default, as in the paper) for one near-miss failed action, one
faithful failed observation, and one structured reflection triplet. Any
OpenAI-compatible chat endpoint works.

Configuration is resolved in this order of precedence:

1. explicit ``model`` / ``api_key`` / ``base_url`` arguments,
2. a ``KEY=VALUE`` file passed via ``env_path`` (optional),
3. process environment variables: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``
   (optional, for self-hosted or proxy endpoints), and ``REGRPO_TEACHER_MODEL``
   (defaults to ``gpt-4o``).

No credentials are ever hard-coded or logged.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

DEFAULT_TEACHER_MODEL = "gpt-4o"


class TeacherError(RuntimeError):
    """Base error for recoverable teacher-generation failures."""


class TeacherConfigError(TeacherError):
    """Raised when teacher API configuration is missing or invalid."""


class TeacherJSONError(TeacherError):
    """Raised when a completion does not contain valid JSON."""


class TeacherAPIError(TeacherError):
    """Raised when the API remains unavailable after retries."""


def load_env_file(env_path: str | Path) -> dict[str, str]:
    """Load settings from a simple ``KEY=VALUE`` file (optional)."""

    path = Path(env_path)
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key.strip()] = value
    return values


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""

    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    text = _strip_json_fence(content)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    block = _first_balanced_object(text)
    if block is None and text != content:
        block = _first_balanced_object(content)
    if block is None:
        raise TeacherJSONError("completion did not contain a balanced JSON object")
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        raise TeacherJSONError(f"completion JSON object did not parse: {exc}") from exc
    if not isinstance(obj, dict):
        raise TeacherJSONError("completion JSON payload is not an object")
    return obj


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline < 0:
        return text
    tail = text[first_newline + 1 :]
    if tail.rstrip().endswith("```"):
        tail = tail.rstrip()[:-3]
    return tail.strip()


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


class TeacherClient:
    """OpenAI SDK wrapper that returns parsed JSON objects from a teacher VLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env_path: str | Path | None = None,
        max_completion_tokens: int = 6000,
        max_retries: int = 5,
        base_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.env_path = env_path
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._client: Any | None = None
        self._model: str | None = None

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Call the teacher model and parse a JSON object from the response."""

        client, model = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self.max_completion_tokens,
                )
                content = response.choices[0].message.content or ""
                return extract_json_object(content)
            except TeacherJSONError:
                raise
            except Exception as exc:
                if not _is_transient_openai_error(exc) or attempt >= self.max_retries:
                    raise TeacherAPIError("teacher completion failed") from exc
                last_error = exc
                sleep_s = self.base_backoff * (2**attempt) + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)
        raise TeacherAPIError("teacher completion failed") from last_error

    def _get_client(self) -> tuple[Any, str]:
        if self._client is not None and self._model is not None:
            return self._client, self._model
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise TeacherConfigError("openai package is required for RoT generation") from exc

        env: dict[str, str] = {}
        if self.env_path is not None:
            env = load_env_file(self.env_path)

        api_key = self.api_key or env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = (
            self.base_url
            or env.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or None
        )
        model = (
            self.model
            or env.get("REGRPO_TEACHER_MODEL")
            or os.environ.get("REGRPO_TEACHER_MODEL")
            or DEFAULT_TEACHER_MODEL
        )
        if not api_key:
            raise TeacherConfigError(
                "missing OpenAI API key: set OPENAI_API_KEY (or pass api_key / env_path)"
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        return self._client, self._model


def _is_transient_openai_error(exc: Exception) -> bool:
    try:
        import openai
    except ImportError:
        return False
    transient_types = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    )
    if isinstance(exc, transient_types):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and 500 <= status_code < 600
