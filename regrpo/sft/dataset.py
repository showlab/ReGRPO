# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""SFT data loading and token-span loss masks for clean and RoT records."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from regrpo.common.io import read_json, read_jsonl
from regrpo.common.schema import _final_response_index, validate_rot_record

LABEL_IGNORE_INDEX = -100
VALID_KINDS = {"clean", "rot"}


def trainable_indices(messages: list[dict], kind: str) -> list[int]:
    """Return assistant-turn indices that should receive SFT loss."""

    _validate_kind(kind)
    if kind == "clean":
        return [idx for idx, turn in enumerate(messages) if turn.get("role") == "assistant"]
    target = _final_response_index(messages)
    return [target] if target >= 0 else []


def build_example(messages: list[dict], tokenizer, kind: str, max_len: int) -> dict:
    """Build one masked SFT example.

    The default truncation window is left-anchored at the full chat sequence
    start. If that would drop a trainable token span, the window is shifted
    right so the complete trainable span stays in the example and only
    non-trainable prefix tokens are left-truncated. If all trainable spans
    cannot fit into ``max_len``, the example fails loudly instead of silently
    training on a partial target.
    """

    _validate_kind(kind)
    if max_len <= 0:
        raise ValueError("max_len must be positive")

    full = _chat_ids(tokenizer, messages)
    labels = [LABEL_IGNORE_INDEX] * len(full)
    spans = [_assistant_span(messages, tokenizer, idx, full) for idx in trainable_indices(messages, kind)]
    for start, end in spans:
        labels[start:end] = full[start:end]

    start, end = _truncation_window(len(full), spans, max_len)
    input_ids = full[start:end]
    label_ids = labels[start:end]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": label_ids,
    }


class RotSFTDataset(torch.utils.data.Dataset):
    """Dataset over a clean/RoT record mix."""

    def __init__(
        self,
        records: list[tuple[dict, str]],
        tokenizer,
        max_len: int,
        clean_ratio: float = 0.5,
        seed: int = 13,
        max_clean: int | None = None,
        max_rot: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.records = _mix_pairs(records, clean_ratio, seed, max_clean=max_clean, max_rot=max_rot)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record, kind = self.records[index]
        messages = _record_messages(record)
        if kind == "rot":
            problems = validate_rot_record(record)
            if problems:
                raise ValueError(f"invalid RoT record {record.get('id', index)}: {problems}")
        return build_example(messages, self.tokenizer, kind, self.max_len)


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad SFT examples; label pads are ignored by CE loss."""

    if not batch:
        raise ValueError("batch must be non-empty")
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [LABEL_IGNORE_INDEX] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_sft_records(
    clean_path: str | Path,
    rot_path: str | Path,
    clean_ratio: float,
    seed: int,
    limit: int | None,
) -> list[tuple[dict, str]]:
    """Load and mix clean MAT records with RoT records."""

    clean = [(record, "clean") for record in _read_records(clean_path)]
    rot = [(record, "rot") for record in _read_records(rot_path, synthesize_rot=True)]
    return _mix_pairs(clean + rot, clean_ratio, seed, limit=limit)


def _validate_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}, got {kind!r}")


def _chat_ids(tokenizer, messages: list[dict], **kwargs: Any) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _assistant_span(
    messages: list[dict],
    tokenizer,
    index: int,
    full: list[int] | None = None,
) -> tuple[int, int]:
    if index < 0 or index >= len(messages):
        raise IndexError(index)
    if messages[index].get("role") != "assistant":
        raise ValueError(f"turn {index} is not an assistant turn")
    full_ids = full if full is not None else _chat_ids(tokenizer, messages)
    prefix = _chat_ids(tokenizer, messages[:index], add_generation_prompt=True)
    if callable(tokenizer):
        content = str(messages[index].get("content", ""))
        content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        if full_ids[len(prefix) : len(prefix) + len(content_ids)] == content_ids:
            return len(prefix), len(prefix) + len(content_ids)
    end = _chat_ids(tokenizer, messages[: index + 1])
    if full_ids[: len(prefix)] == prefix and full_ids[: len(end)] == end:
        return len(prefix), len(end)

    # Some tokenizer/template versions may not keep the generation prompt as a
    # strict full-render prefix. Fall back to locating the assistant turn tokens
    # inside the full render and fail if the match is ambiguous.
    target = end[len(prefix) :] if len(end) >= len(prefix) else []
    if not target:
        content = str(messages[index].get("content", ""))
        target = tokenizer(content, add_special_tokens=False)["input_ids"]
    matches = _subsequence_matches(full_ids, target)
    if len(matches) != 1:
        raise AssertionError(
            f"chat-template prefix invariant failed and fallback found {len(matches)} matches"
        )
    start = matches[0]
    return start, start + len(target)


def _subsequence_matches(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    width = len(needle)
    return [
        idx
        for idx in range(0, len(haystack) - width + 1)
        if haystack[idx : idx + width] == needle
    ]


def _truncation_window(
    full_len: int,
    spans: list[tuple[int, int]],
    max_len: int,
) -> tuple[int, int]:
    if full_len <= max_len:
        return 0, full_len
    if not spans:
        return 0, max_len
    first = min(start for start, _ in spans)
    last = max(end for _, end in spans)
    if last - first > max_len:
        raise ValueError("trainable token spans do not fit within max_len")
    start = 0 if last <= max_len else last - max_len
    if first < start:
        start = first
    end = start + max_len
    if last > end:
        end = last
        start = end - max_len
    if any(span_start < start or span_end > end for span_start, span_end in spans):
        raise ValueError("truncation would drop a trainable token span")
    return start, min(end, full_len)


def _record_messages(record: dict) -> list[dict]:
    messages = record.get("conversations") or record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record must contain conversations or messages")
    return messages


def _read_records(path: str | Path, synthesize_rot: bool = False) -> list[dict]:
    source = Path(path)
    if not source.exists():
        if synthesize_rot:
            return _synthetic_rot_records()
        raise FileNotFoundError(source)
    if source.suffix == ".jsonl":
        return list(read_jsonl(source))
    data = read_json(source)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{source} must contain a JSON object or array")


def _mix_pairs(
    records: list[tuple[dict, str]],
    clean_ratio: float,
    seed: int,
    limit: int | None = None,
    max_clean: int | None = None,
    max_rot: int | None = None,
) -> list[tuple[dict, str]]:
    if not 0.0 <= clean_ratio <= 1.0:
        raise ValueError("clean_ratio must be in [0, 1]")
    rng = random.Random(seed)
    clean = [item for item in records if item[1] == "clean"]
    rot = [item for item in records if item[1] == "rot"]
    rng.shuffle(clean)
    rng.shuffle(rot)
    if max_clean is not None:
        clean = clean[: max(0, int(max_clean))]
    if max_rot is not None:
        rot = rot[: max(0, int(max_rot))]

    if limit is not None:
        total = max(0, int(limit))
        clean_n = min(len(clean), int(round(total * clean_ratio)))
        rot_n = min(len(rot), total - clean_n)
        remaining = total - clean_n - rot_n
        if remaining > 0:
            clean_take = min(len(clean) - clean_n, remaining)
            clean_n += clean_take
            remaining -= clean_take
        if remaining > 0:
            rot_n += min(len(rot) - rot_n, remaining)
    elif clean_ratio <= 0:
        clean_n, rot_n = 0, len(rot)
    elif clean_ratio >= 1:
        clean_n, rot_n = len(clean), 0
    elif clean and rot:
        clean_from_rot = int(round(len(rot) * clean_ratio / (1.0 - clean_ratio)))
        clean_n = min(len(clean), max(1, clean_from_rot))
        rot_from_clean = int(round(clean_n * (1.0 - clean_ratio) / clean_ratio))
        rot_n = min(len(rot), max(1, rot_from_clean))
    else:
        clean_n, rot_n = len(clean), len(rot)

    mixed = clean[:clean_n] + rot[:rot_n]
    rng.shuffle(mixed)
    return mixed


def _synthetic_rot_records() -> list[dict]:
    records = []
    for idx in range(3):
        records.append(
            {
                "id": f"synthetic_rot_{idx}",
                "image": {},
                "answer": "done",
                "conversations": [
                    {"role": "system", "content": "Use tools."},
                    {"role": "user", "content": "Task: answer with done."},
                    {
                        "role": "assistant",
                        "content": "Thought: bad path\nCode:\n```py\nprint('bad')\n```",
                    },
                    {
                        "role": "user",
                        "content": "[OUTPUT OF STEP 0] -> Observation:\nError: bad path failed",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Reflection:\n"
                            "- ErrorType: ArgInvalid\n"
                            "- Evidence: The observation says bad path failed.\n"
                            "- FixPlan: Use the direct final answer.\n"
                            "Thought: answer directly\n"
                            "Code:\n```py\nfinal_answer('done')\n```"
                        ),
                    },
                ],
                "rot_meta": {
                    "src_id": f"synthetic_{idx}",
                    "step_index": 0,
                    "error_type": "ArgInvalid",
                    "failed_action": "Thought: bad path\nCode:\n```py\nprint('bad')\n```",
                    "failed_observation": "Error: bad path failed",
                    "reflection": {
                        "error_type": "ArgInvalid",
                        "evidence": "The observation says bad path failed.",
                        "fix_plan": "Use the direct final answer.",
                    },
                    "corrected_action": "Thought: answer directly\nCode:\n```py\nfinal_answer('done')\n```",
                    "final_answer": "done",
                },
            }
        )
    return records
