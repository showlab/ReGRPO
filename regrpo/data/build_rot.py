# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Build teacher-generated RoT data from MAT trajectories."""

from __future__ import annotations

import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from regrpo.common.io import append_jsonl, completed_ids, iter_json_array, read_jsonl, write_json
from regrpo.common.trajectory import parse_trajectory
from regrpo.data.teacher_client import TeacherClient, TeacherError
from regrpo.data.perturb import perturbation_plan
from regrpo.data.prompts import build_generation_prompt, build_repair_prompt
from regrpo.data.quality import _format_action, assemble_rot_record, check_generation


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.input:
        config["input_path"] = args.input
    if args.output:
        config["output_path"] = args.output
    stats = build_rot(config, limit=args.limit)
    print(
        "done "
        f"ok={stats['ok']} repaired={stats['repaired']} discarded={stats['discarded']} "
        f"written={stats['written']} output={config['output_path']}"
    )


def build_rot(config: dict[str, Any], limit: int | None = None) -> dict[str, int]:
    seed = int(config.get("seed", 13))
    rng = random.Random(seed)
    max_records = _resolve_limit(config.get("num_trajectories"), limit)
    records = list(iter_json_array(config["input_path"]))
    if max_records is not None and max_records < len(records):
        records = rng.sample(records, max_records)

    checkpoint_path = Path(config["checkpoint_path"])
    done_ids = completed_ids(checkpoint_path)
    lock = threading.Lock()
    stats = {"ok": 0, "repaired": 0, "discarded": 0, "written": 0}
    samples_per_traj = int(config.get("samples_per_traj", 1))
    max_retries = int(config.get("max_retries", 3))
    concurrency = int(config.get("concurrency", 1))
    enable_non_commit = bool(config.get("enable_non_commit", False))
    teacher_cfg = config.get("teacher") or {}
    teacher_model = teacher_cfg.get("model")
    env_path = teacher_cfg.get("env_path")

    jobs = []
    for record_index, record in enumerate(records):
        try:
            traj = parse_trajectory(record)
            plan = perturbation_plan(
                traj,
                random.Random(seed + record_index),
                samples_per_traj,
                enable_non_commit=enable_non_commit,
            )
        except Exception as exc:
            print(f"discard source={record.get('id')} reason=parse_or_plan_error:{exc}")
            stats["discarded"] += 1
            continue
        for step, error_type in plan:
            rot_id = f"{traj.src_id}_rot_s{step.index}_{error_type.value}"
            if rot_id in done_ids:
                continue
            jobs.append((traj, step, error_type))

    def write_record(obj: dict[str, Any]) -> None:
        with lock:
            append_jsonl(checkpoint_path, obj)
            done_ids.add(str(obj["id"]))
            stats["written"] += 1

    def add_stat(key: str) -> None:
        with lock:
            stats[key] += 1
            print(
                f"tally ok={stats['ok']} repaired={stats['repaired']} "
                f"discarded={stats['discarded']} written={stats['written']}"
            )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(_process_one, job, env_path, teacher_model, max_retries, write_record)
            for job in jobs
        ]
        for future in as_completed(futures):
            outcome = future.result()
            add_stat(outcome)

    _collate_checkpoint(checkpoint_path, config["output_path"])
    return stats


def _process_one(
    job: tuple[Any, Any, Any],
    env_path: str | None,
    teacher_model: str | None,
    max_retries: int,
    write_record,
) -> str:
    traj, step, error_type = job
    client = TeacherClient(model=teacher_model, env_path=env_path)
    try:
        corrected = _format_action(step)
        system, user = build_generation_prompt(traj, step, error_type)
        gen = client.complete_json(system, user)
        repaired = False
        for _ in range(max_retries + 1):
            problems = check_generation(gen, error_type, corrected_action=corrected)
            if not problems:
                record = assemble_rot_record(traj, step, error_type, gen)
                write_record(record.to_dict())
                return "repaired" if repaired else "ok"
            if repaired and _ >= max_retries:
                break
            system, user = build_repair_prompt(gen, problems, error_type)
            gen = client.complete_json(system, user)
            repaired = True
        print(f"discard source={traj.src_id} step={step.index} reason=quality:{problems}")
    except (TeacherError, ValueError, KeyError, TypeError) as exc:
        print(f"discard source={traj.src_id} step={step.index} reason={type(exc).__name__}:{exc}")
    except Exception as exc:
        print(f"discard source={traj.src_id} step={step.index} reason=unexpected:{exc}")
    return "discarded"


def _collate_checkpoint(checkpoint_path: Path, output_path: str | Path) -> None:
    rows = list(read_jsonl(checkpoint_path)) if checkpoint_path.exists() else []
    rows.sort(key=lambda obj: str(obj.get("id", "")))
    write_json(output_path, rows)


def _resolve_limit(config_value: Any, cli_limit: int | None) -> int | None:
    if cli_limit is not None:
        return cli_limit
    if config_value is None:
        return None
    if isinstance(config_value, str) and config_value.lower() == "all":
        return None
    return int(config_value)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input", help="Override input_path (source clean trajectories)")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    main()
