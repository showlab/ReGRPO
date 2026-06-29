# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Perturbation taxonomy and deterministic sampling."""

from __future__ import annotations

import random

from regrpo.common.schema import ErrorType
from regrpo.common.trajectory import (
    TERMINAL_TOOL,
    Step,
    Trajectory,
    extract_tool_calls,
    perturbable_steps,
)

PAPER_DISTRIBUTION: dict[ErrorType, float] = {
    ErrorType.GROUNDING_DRIFT: 0.294,
    ErrorType.TOOL_MISMATCH: 0.246,
    ErrorType.ARG_INVALID: 0.308,
    ErrorType.INFO_INSUFFICIENT: 0.152,
}

# GroundingDrift requires a vision/spatial tool; InfoInsufficient is restricted
# to search/file tools where dropped necessary context is verifiable.
FEASIBILITY: dict[str, frozenset[ErrorType]] = {
    "ask_search_agent": frozenset(
        {
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
            ErrorType.INFO_INSUFFICIENT,
        }
    ),
    "inspect_file_as_text": frozenset(
        {
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
            ErrorType.INFO_INSUFFICIENT,
        }
    ),
    "visualizer": frozenset(ErrorType),
    "objectlocation": frozenset(
        {
            ErrorType.GROUNDING_DRIFT,
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
        }
    ),
    "image_edit": frozenset(
        {
            ErrorType.GROUNDING_DRIFT,
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
        }
    ),
    "facedetection": frozenset(
        {
            ErrorType.GROUNDING_DRIFT,
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
        }
    ),
    "segmentation": frozenset(
        {
            ErrorType.GROUNDING_DRIFT,
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
        }
    ),
    "image_generator": frozenset(
        {
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
        }
    ),
    "search": frozenset(
        {
            ErrorType.TOOL_MISMATCH,
            ErrorType.ARG_INVALID,
            ErrorType.INFO_INSUFFICIENT,
        }
    ),
    TERMINAL_TOOL: frozenset(),
}


def feasible_error_types(step: Step) -> frozenset[ErrorType]:
    """Return feasible error types for a parsed step."""

    if step.primary_tool is None:
        return frozenset()
    return FEASIBILITY.get(step.primary_tool, frozenset())


def sample_error_type(step: Step, rng: random.Random) -> ErrorType | None:
    """Sample one feasible error type from renormalized paper weights."""

    feasible = feasible_error_types(step)
    if not feasible:
        return None
    ordered = [error_type for error_type in PAPER_DISTRIBUTION if error_type in feasible]
    total = sum(PAPER_DISTRIBUTION[error_type] for error_type in ordered)
    draw = rng.random() * total
    cumulative = 0.0
    for error_type in ordered:
        cumulative += PAPER_DISTRIBUTION[error_type]
        if draw <= cumulative:
            return error_type
    return ordered[-1]


def final_commit_step(traj: Trajectory) -> Step | None:
    """Return the trajectory's clean final_answer step, if any.

    NonCommit targets this step: the corrected action is its concise
    final_answer(...) call. We require the step to actually parse a
    final_answer tool call so the corrected action is genuinely committal.
    """

    if not traj.steps:
        return None
    step = traj.steps[-1]
    if not step.is_final:
        return None
    calls = extract_tool_calls(step.code)
    if not any(call.name == TERMINAL_TOOL for call in calls):
        return None
    return step


def perturbation_plan(
    traj: Trajectory,
    rng: random.Random,
    samples_per_traj: int = 1,
    enable_non_commit: bool = False,
) -> list[tuple[Step, ErrorType]]:
    """Pick perturbable steps and assign deterministic sampled error types.

    When ``enable_non_commit`` is True, a single NonCommit job targeting the
    trajectory's final_answer step is appended. The default (False) keeps the
    plan byte-identical to the original sampling so existing generation is
    unaffected.
    """

    if samples_per_traj <= 0:
        return []
    plan: list[tuple[Step, ErrorType]] = []
    candidates = perturbable_steps(traj)
    if candidates:
        count = min(samples_per_traj, len(candidates))
        selected = rng.sample(candidates, count)
        for step in selected:
            error_type = sample_error_type(step, rng)
            if error_type is not None:
                plan.append((step, error_type))
    if enable_non_commit:
        final = final_commit_step(traj)
        if final is not None:
            plan.append((final, ErrorType.NON_COMMIT))
    return plan
