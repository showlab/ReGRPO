# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Small JSON and JSONL helpers for deterministic data generation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Write a UTF-8 JSON file, creating parent directories as needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield JSONL objects, skipping blank lines."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def append_jsonl(path: str | Path, obj: Any) -> None:
    """Append one compact JSON object line."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def completed_ids(jsonl_path: str | Path, id_key: str = "id") -> set[str]:
    """Return completed ids from a checkpoint JSONL file.

    Missing files return an empty set. Malformed trailing lines are ignored to
    support crash resume after partial writes.
    """

    source = Path(jsonl_path)
    if not source.exists():
        return set()
    done: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            value = obj.get(id_key)
            if value is not None:
                done.add(str(value))
    return done


def iter_json_array(path: str | Path) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without loading it all at once."""

    decoder = json.JSONDecoder()
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_array = False
        done = False
        while not done:
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                done = True
            while True:
                buffer = buffer.lstrip()
                if not in_array:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{source} is not a top-level JSON array")
                    buffer = buffer[1:]
                    in_array = True
                    continue
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if not buffer:
                    break
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if done:
                        raise
                    break
                if not isinstance(obj, dict):
                    raise ValueError(f"{source} contains a non-object array item")
                yield obj
                buffer = buffer[end:]
