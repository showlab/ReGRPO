# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Convert clean/RoT records into MAT Qwen-VL finetuning JSON."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from regrpo.common.io import read_json, read_jsonl, write_json
from regrpo.common.schema import _final_response_index

VALID_KINDS = {"clean", "rot"}


def convert_record(record: dict, kind: str) -> dict:
    """Convert one clean or RoT record into MAT Qwen-VL conversation format."""

    _validate_kind(kind)
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("record must contain a conversations list")

    image_map = _image_path_map(record.get("image"))
    schema_conversations = [_schema_turn(turn) for turn in conversations]
    converted = {
        "id": str(record.get("id", "")),
        "image": dict(image_map),
        "conversations": [_convert_turn(turn, image_map) for turn in conversations],
        "mask_policy": kind,
    }
    if kind == "rot":
        target = _final_response_index(schema_conversations)
        if target < 0:
            raise ValueError(f"RoT record {converted['id']!r} has no assistant target turn")
        converted["train_turn_index"] = target
    return converted


def build_qwen_vl_dataset(
    rot_path: str | Path,
    clean_path: str | Path,
    clean_ratio: float,
    seed: int,
    limit: int | None,
    out_path: str | Path,
) -> list[dict]:
    """Load, convert, mix, and write a Qwen-VL SFT JSON array."""

    records: list[tuple[dict, str]] = []
    records.extend((convert_record(record, "clean"), "clean") for record in _read_records(clean_path))
    records.extend((convert_record(record, "rot"), "rot") for record in _read_records(rot_path))
    mixed = _mix_pairs(records, clean_ratio, seed, limit=limit)
    output = [record for record, _ in mixed]
    write_json(out_path, output)
    counts = Counter(item["mask_policy"] for item in output)
    print(
        "[convert_qwen_vl] wrote "
        f"{len(output)} records to {out_path} "
        f"(clean={counts.get('clean', 0)} rot={counts.get('rot', 0)})"
    )
    return output


def _mix_pairs(
    records: list[tuple[dict, str]],
    clean_ratio: float,
    seed: int,
    limit: int | None = None,
) -> list[tuple[dict, str]]:
    """Mix clean/RoT pairs using the same ratio semantics as the text SFT path."""

    if not 0.0 <= clean_ratio <= 1.0:
        raise ValueError("clean_ratio must be in [0, 1]")
    rng = random.Random(seed)
    clean = [item for item in records if item[1] == "clean"]
    rot = [item for item in records if item[1] == "rot"]
    rng.shuffle(clean)
    rng.shuffle(rot)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rot", required=True, help="Path to rot_train.json/jsonl")
    parser.add_argument("--clean", required=True, help="Path to mat_train.json/jsonl")
    parser.add_argument("--clean-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output Qwen-VL JSON path")
    args = parser.parse_args(argv)
    build_qwen_vl_dataset(
        rot_path=args.rot,
        clean_path=args.clean,
        clean_ratio=args.clean_ratio,
        seed=args.seed,
        limit=args.limit,
        out_path=args.out,
    )


def _validate_kind(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}, got {kind!r}")


def _read_records(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return list(read_jsonl(source))
    data = read_json(source)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{source} must contain a JSON object or array")


def _image_path_map(image: Any) -> dict[str, str]:
    if image is None or image == "":
        return {}
    if isinstance(image, str):
        return {"<image>": image}
    if isinstance(image, dict):
        return {str(key): str(value) for key, value in image.items()}
    raise ValueError(f"unsupported image field type: {type(image).__name__}")


def _convert_turn(turn: dict, image_map: dict[str, str]) -> dict:
    role = turn.get("role", turn.get("from"))
    content = str(turn.get("content", turn.get("value", "")))
    for pid, key in enumerate(sorted(image_map), start=1):
        if key in content:
            content = content.replace(key, f"Picture {pid}: <img>{image_map[key]}</img>\n")
            content = content.replace("</img>\n\n", "</img>\n")
    return {"from": str(role), "value": content}


def _schema_turn(turn: dict) -> dict:
    return {
        "role": turn.get("role", turn.get("from")),
        "content": str(turn.get("content", turn.get("value", ""))),
    }


if __name__ == "__main__":
    main()
