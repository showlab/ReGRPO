# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Parse MAT trajectories into typed offline steps."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

# Grounded in observed source call sites; later dispatches can extend it.
TOOL_REGISTRY: frozenset[str] = frozenset(
    {
        "ask_search_agent",
        "visualizer",
        "inspect_file_as_text",
        "image_generator",
        "objectlocation",
        "image_edit",
        "facedetection",
        "segmentation",
        "search",
        "final_answer",
    }
)
TERMINAL_TOOL = "final_answer"
CODE_BLOCK_RE = re.compile(r"```(?:py|python)?\s*\n(.*?)```", re.DOTALL)
OBS_PREFIX_RE = re.compile(r"^\[OUTPUT OF STEP\s+\d+\]\s*->\s*Observation:\s*", re.DOTALL)


def _system_role() -> str:
    return "system"


def _user_role() -> str:
    return "user"


def _response_role() -> str:
    return "assistant"


@dataclass(frozen=True)
class ToolCall:
    """A parsed agent tool call."""

    name: str
    n_args: int
    kw_keys: tuple[str, ...]
    lineno: int
    first_arg_repr: str | None = None


@dataclass
class Step:
    """One thought, code action, optional observation tuple."""

    index: int
    thought: str
    code: str
    tool_calls: list[ToolCall]
    primary_tool: str | None
    is_final: bool
    observation: str | None


@dataclass
class Trajectory:
    """Parsed source record trajectory."""

    src_id: str
    system: str
    task: str
    steps: list[Step]
    answer: Any
    image: Any


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _safe_unparse(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _first_arg_repr(node: ast.Call) -> str | None:
    if node.args:
        return _safe_unparse(node.args[0])
    for key in ("file_path", "question", "query"):
        for keyword in node.keywords:
            if keyword.arg == key:
                return _safe_unparse(keyword.value)
    return None


def extract_tool_calls(code: str) -> list[ToolCall]:
    """Extract known agent tool calls from Python code in source order."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls: list[ToolCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in TOOL_REGISTRY:
            continue
        calls.append(
            ToolCall(
                name=name,
                n_args=len(node.args),
                kw_keys=tuple(keyword.arg for keyword in node.keywords if keyword.arg),
                lineno=getattr(node, "lineno", 0),
                first_arg_repr=_first_arg_repr(node),
            )
        )
    return sorted(calls, key=lambda call: call.lineno)


def _split_turn(content: str) -> tuple[str, str]:
    match = CODE_BLOCK_RE.search(content)
    if not match:
        return content.strip(), ""
    thought = content[: match.start()].strip()
    # Source turns are "Thought: ...\n\nCode:\n```...```"; everything before the
    # fence keeps the structural "Code:" marker. Strip it so `thought` holds only
    # the reasoning text and reconstruction does not duplicate the marker.
    thought = re.sub(r"\s*Code\s*:\s*$", "", thought).strip()
    return thought, match.group(1).strip()


def _clean_observation(content: str) -> str:
    """Strip the source observation prefix and keep the payload text."""

    return OBS_PREFIX_RE.sub("", content.strip()).strip()


def _primary_tool(tool_calls: list[ToolCall]) -> str | None:
    for call in reversed(tool_calls):
        if call.name != TERMINAL_TOOL:
            return call.name
    if any(call.name == TERMINAL_TOOL for call in tool_calls):
        return TERMINAL_TOOL
    return None


def parse_trajectory(record: dict) -> Trajectory:
    """Parse a raw MAT record into a trajectory."""

    conversations = record.get("conversations") or []
    system = ""
    task = ""
    steps: list[Step] = []
    for index, turn in enumerate(conversations):
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == _system_role() and not system:
            system = content
        elif role == _user_role() and not task and not content.startswith("[OUTPUT OF STEP"):
            task = content
        elif role == _response_role():
            thought, code = _split_turn(content)
            calls = extract_tool_calls(code)
            observation = None
            if index + 1 < len(conversations):
                next_turn = conversations[index + 1]
                if next_turn.get("role") == _user_role():
                    next_content = next_turn.get("content") or ""
                    if next_content.startswith("[OUTPUT OF STEP"):
                        observation = _clean_observation(next_content)
            steps.append(
                Step(
                    index=len(steps),
                    thought=thought,
                    code=code,
                    tool_calls=calls,
                    primary_tool=_primary_tool(calls),
                    is_final=any(call.name == TERMINAL_TOOL for call in calls),
                    observation=observation,
                )
            )
    return Trajectory(
        src_id=str(record.get("id", "")),
        system=system,
        task=task,
        steps=steps,
        answer=record.get("answer"),
        image=record.get("image"),
    )


def perturbable_steps(traj: Trajectory) -> list[Step]:
    """Return non-final steps with a perturbable primary agent tool."""

    tools = TOOL_REGISTRY - {TERMINAL_TOOL}
    return [
        step
        for step in traj.steps
        if not step.is_final and step.tool_calls and step.primary_tool in tools
    ]
