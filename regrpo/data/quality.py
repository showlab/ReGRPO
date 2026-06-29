# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Quality gates and assembly for generated RoT records."""

from __future__ import annotations

import ast
from typing import Any

from regrpo.common.schema import (
    ErrorType,
    Reflection,
    RotMeta,
    RotRecord,
    evidence_grounded,
    has_label_leak,
    validate_rot_record,
)
from regrpo.common.trajectory import TERMINAL_TOOL, Step, Trajectory, extract_tool_calls


def check_generation(
    gen: dict,
    error_type: ErrorType,
    corrected_action: str | None = None,
) -> list[str]:
    """Validate a raw GPT generation before RoT assembly."""

    problems: list[str] = []
    for key in ("failed_action", "failed_observation", "reflection"):
        if key not in gen:
            problems.append(f"missing key: {key}")
    if problems:
        return problems

    failed_action = gen.get("failed_action")
    failed_observation = gen.get("failed_observation")
    reflection = gen.get("reflection")
    if not isinstance(failed_action, str) or not failed_action.strip():
        problems.append("failed_action must be a non-empty string")
    elif "Thought:" not in failed_action or "Code:" not in failed_action:
        problems.append("failed_action must contain Thought and Code")
    elif error_type is ErrorType.NON_COMMIT:
        # NonCommit's failure IS the missing commit: the code must be valid
        # Python but must NOT call final_answer (the whole point is it never
        # commits the answer). It may legitimately have no recognized tool call.
        failed_code = _extract_code_for_parse(failed_action)
        if not _is_parsable_python(failed_code):
            problems.append("failed_action code must be valid Python")
        elif _calls_tool(failed_code, TERMINAL_TOOL):
            problems.append("NonCommit failed_action must NOT call final_answer")
    elif not extract_tool_calls(_extract_code_for_parse(failed_action)):
        problems.append("failed_action must contain a parsable tool call")

    if not isinstance(failed_observation, str) or not failed_observation.strip():
        problems.append("failed_observation must be a non-empty string")

    if not isinstance(reflection, dict):
        problems.append("reflection must be an object")
        return problems
    for key in ("error_type", "evidence", "fix_plan"):
        if key not in reflection:
            problems.append(f"missing reflection key: {key}")
    if reflection.get("error_type") != error_type.value:
        problems.append(
            f"reflection.error_type must be {error_type.value}, got {reflection.get('error_type')}"
        )
    evidence = reflection.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        problems.append("reflection.evidence must be a non-empty string")
    elif isinstance(failed_observation, str) and not evidence_grounded(evidence, failed_observation):
        problems.append("reflection.evidence is not grounded in failed_observation")
    if not isinstance(reflection.get("fix_plan"), str) or not reflection.get("fix_plan", "").strip():
        problems.append("reflection.fix_plan must be a non-empty string")
    if (
        error_type is ErrorType.TOOL_MISMATCH
        and corrected_action is not None
        and isinstance(failed_action, str)
    ):
        failed_tool = _primary_agent_tool(failed_action)
        corrected_tool = _primary_agent_tool(corrected_action)
        if failed_tool is None or failed_tool == corrected_tool:
            problems.append(
                "ToolMismatch failed_action must call a different tool than the corrected action"
            )
    return problems


def assemble_rot_record(
    traj: Trajectory,
    step: Step,
    error_type: ErrorType,
    gen: dict[str, Any],
) -> RotRecord:
    """Assemble a hybrid RoT record from a checked generation."""

    failed_action = str(gen["failed_action"]).strip()
    failed_observation = str(gen["failed_observation"]).strip()
    reflection_obj = gen["reflection"]
    reflection = Reflection(
        error_type=error_type,
        evidence=str(reflection_obj["evidence"]).strip(),
        fix_plan=str(reflection_obj["fix_plan"]).strip(),
    )
    corrected_action = _format_action(step)
    conversations = [
        {"role": "system", "content": traj.system},
        {"role": "user", "content": traj.task},
    ]
    for prev in traj.steps[: step.index]:
        conversations.append({"role": "assistant", "content": _format_action(prev)})
        if prev.observation is not None:
            conversations.append(
                {
                    "role": "user",
                    "content": f"[OUTPUT OF STEP {prev.index}] -> Observation:\n{prev.observation}",
                }
            )
    conversations.extend(
        [
            {"role": "assistant", "content": failed_action},
            {
                "role": "user",
                "content": f"[OUTPUT OF STEP {step.index}] -> Observation:\n{failed_observation}",
            },
            {"role": "assistant", "content": _reflection_action(reflection, corrected_action)},
        ]
    )
    if step.observation is not None:
        conversations.append(
            {
                "role": "user",
                "content": f"[OUTPUT OF STEP {step.index}] -> Observation:\n{step.observation}",
            }
        )
    for later in traj.steps[step.index + 1 :]:
        conversations.append({"role": "assistant", "content": _format_action(later)})
        if later.observation is not None:
            conversations.append(
                {
                    "role": "user",
                    "content": f"[OUTPUT OF STEP {later.index}] -> Observation:\n{later.observation}",
                }
            )

    meta = RotMeta(
        src_id=traj.src_id,
        step_index=step.index,
        error_type=error_type,
        failed_action=failed_action,
        failed_observation=failed_observation,
        reflection=reflection,
        corrected_action=corrected_action,
        final_answer=str(traj.answer),
    )
    record = RotRecord(
        id=f"{traj.src_id}_rot_s{step.index}_{error_type.value}",
        image=traj.image,
        answer=traj.answer,
        conversations=conversations,
        rot_meta=meta,
    )
    problems = validate_rot_record(record)
    if has_label_leak(record.conversations, corrected_action):
        problems.append("corrected_action leaks before target turn")
    # The failed action must actually perturb a*: an identical Code body means no
    # perturbation happened (the "failure" would live only in the synthetic obs).
    failed_code = _extract_code_for_parse(failed_action)
    if failed_code and failed_code == _extract_code_for_parse(corrected_action):
        problems.append("failed_action code is identical to the corrected action")
    if problems:
        raise ValueError("; ".join(problems))
    return record


def _extract_code_for_parse(action: str) -> str:
    marker = "```"
    if marker not in action:
        return action
    start = action.find(marker)
    code_start = action.find("\n", start)
    if code_start < 0:
        return action
    end = action.find(marker, code_start + 1)
    if end < 0:
        return action[code_start + 1 :]
    return action[code_start + 1 : end].strip()


def _is_parsable_python(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _calls_tool(code: str, tool_name: str) -> bool:
    """Return whether code contains a call to ``tool_name`` (any code, even
    those with no other recognized tool call)."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == tool_name:
            return True
    return False


def _primary_agent_tool(action: str) -> str | None:
    calls = extract_tool_calls(_extract_code_for_parse(action))
    for call in reversed(calls):
        if call.name != TERMINAL_TOOL:
            return call.name
    return None


def _format_action(step: Step) -> str:
    return f"{step.thought.strip()}\n\nCode:\n```py\n{step.code.strip()}\n```".strip()


def _reflection_action(reflection: Reflection, corrected_action: str) -> str:
    return (
        "Reflection:\n"
        f"- ErrorType: {reflection.error_type.value}\n"
        f"- Evidence: {reflection.evidence}\n"
        f"- FixPlan: {reflection.fix_plan}\n"
        f"{corrected_action}"
    )
