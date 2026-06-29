# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Prompt builders for teacher-VLM RoT generation."""

from __future__ import annotations

import json

from regrpo.common.schema import ErrorType
from regrpo.common.trajectory import TOOL_REGISTRY, Step, Trajectory

SYSTEM_PROMPT = """You are a strict data generation engine for ReGRPO.
Produce one realistic near-miss failed action, one faithful failed observation, and one Reflection-of-Thought triple.
Return STRICT JSON only. Do not include markdown, prose, comments, or extra keys.
The failed action must keep the same Thought plus Code shape as the provided correct action, but it must be broken according to the requested ErrorType.
The failed observation must look like the named tool's real output format and must be consistent with the failed action.
The reflection evidence must quote a concrete token or phrase from failed_observation."""

PER_TOOL_OBS_TEMPLATES: dict[str, str] = {
    "inspect_file_as_text": (
        "Use a file-reading failure: file not found, page out of range, unsupported file, "
        "or empty extracted text. Keep it as plain text returned by the tool."
    ),
    "ask_search_agent": (
        "Use a search failure: no results, timeout, or irrelevant snippets. Return a short "
        "search-agent style string or list of snippets."
    ),
    "search": (
        "Use a search failure: no results, timeout, or irrelevant snippets. Return a short "
        "search result string or list of snippets."
    ),
    "visualizer": (
        "Use a VQA failure: irrelevant answer, empty answer, or a wrong-region description. "
        "Return a concise natural-language visual answer."
    ),
    "objectlocation": (
        "Use an object-location failure: [] for no boxes or boxes for the wrong object. "
        "Return a Python-like bbox list."
    ),
    "segmentation": "Use a segmentation failure: empty mask, null mask path, or no target pixels found.",
    "image_edit": "Use an image-edit failure: size mismatch, invalid format, missing source image, or edit rejected.",
    "image_generator": "Use an image-generation failure: size mismatch, invalid format, unsupported prompt, or empty image path.",
    "facedetection": "Use a face-detection failure: no faces detected or an empty bbox list.",
}

ERROR_TYPE_GUIDANCE: dict[ErrorType, str] = {
    ErrorType.GROUNDING_DRIFT: "Shift the referenced region, object, crop, or visual target while preserving the tool shape.",
    ErrorType.TOOL_MISMATCH: (
        "Replace the tool with a DIFFERENT, unsuitable tool from the toolbox for the same intent. "
        "The failed action MUST call a different tool than target_tool (the correct tool) -- do NOT "
        "keep target_tool. Keep the Thought+Code SHAPE (a Thought then a fenced Code block)."
    ),
    ErrorType.ARG_INVALID: "Corrupt an argument such as a path, page, bbox, object name, query, size, or format.",
    ErrorType.INFO_INSUFFICIENT: "Drop or underspecify necessary context so the tool cannot retrieve enough information.",
    ErrorType.NON_COMMIT: (
        "The agent already has enough to answer but FAILS TO COMMIT. Produce a verbose, rambling "
        "failed action that re-explains/derives the answer in prose and Python (print, reassign "
        "variables, list caveats) but DOES NOT call final_answer(). The Code block MUST be valid "
        "Python and MUST NOT contain a final_answer(...) call -- it should at most print or store "
        "the value. Do not copy the correct final_answer(...) line verbatim anywhere."
    ),
}

# NonCommit's failure is a missed termination contract, not a tool error, so it
# uses a dedicated observation template instead of a per-tool one.
NON_COMMIT_OBS_TEMPLATE = (
    "Emit a runtime/episode note (NOT a tool output) stating the episode did not terminate because "
    "no final_answer call was captured, and echo the verbose answer text the agent produced. "
    'Example shape: \'Episode did not terminate: no final_answer call was captured. '
    "Last emitted answer text: \"<the verbose value>\".'"
)


def build_generation_prompt(traj: Trajectory, step: Step, error_type: ErrorType) -> tuple[str, str]:
    """Build a teacher-view generation prompt for one perturbation."""

    is_non_commit = error_type is ErrorType.NON_COMMIT
    tool = step.primary_tool or "unknown"
    if is_non_commit:
        obs_template = NON_COMMIT_OBS_TEMPLATE
        failed_action_hint = (
            "<Thought+Code that derives/states the answer verbosely but contains NO "
            "final_answer() call (e.g. print or variable assignment only)>"
        )
        failed_obs_hint = "<episode-did-not-terminate note that echoes the verbose answer text>"
    else:
        obs_template = PER_TOOL_OBS_TEMPLATES.get(
            tool, "Use a realistic tool failure in the tool's native output style."
        )
        failed_action_hint = "<Thought+Code, same shape as correct_action_teacher_only but broken>"
        failed_obs_hint = "<realistic failure observation in the target tool output format>"
    user = {
        "task": traj.task,
        "source_id": traj.src_id,
        "step_index": step.index,
        "target_tool": tool,
        "requested_error_type": error_type.value,
        "error_type_guidance": ERROR_TYPE_GUIDANCE[error_type],
        "failure_observation_template": obs_template,
        "history_before_step": _history_before_step(traj, step.index),
        "correct_action_teacher_only": _format_action(step),
        "output_schema": {
            "failed_action": failed_action_hint,
            "failed_observation": failed_obs_hint,
            "reflection": {
                "error_type": error_type.value,
                "evidence": "<quote a token or phrase from failed_observation>",
                "fix_plan": "<concrete corrective strategy>",
            },
        },
    }
    if is_non_commit:
        user["forbidden_in_failed_action"] = "final_answer"
        user["expected_final_answer_teacher_only"] = str(traj.answer)
    if error_type is ErrorType.TOOL_MISMATCH:
        user["forbidden_tool"] = tool
        user["alternative_tools"] = sorted(TOOL_REGISTRY - {tool, "final_answer"})
    return SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False, indent=2)


def build_repair_prompt(
    prev_json: dict,
    problems: list[str],
    error_type: ErrorType | None = None,
) -> tuple[str, str]:
    """Build a repair prompt from validator complaints."""

    if error_type is ErrorType.NON_COMMIT:
        failed_action_schema = (
            "<non-empty Thought+Code that states the answer verbosely but contains NO "
            "final_answer() call>"
        )
    else:
        failed_action_schema = (
            "<non-empty Thought+Code with a parsable tool call; for ToolMismatch the Code "
            "must call a different tool than the correct action>"
        )
    user = {
        "instruction": "Repair the JSON object so it passes every validator complaint. Return STRICT JSON only.",
        "validator_complaints": problems,
        "previous_json": prev_json,
        "required_schema": {
            "failed_action": failed_action_schema,
            "failed_observation": "<non-empty realistic failed observation>",
            "reflection": {
                "error_type": "<must match the requested ErrorType>",
                "evidence": "<must quote a concrete token from failed_observation>",
                "fix_plan": "<concrete corrective strategy>",
            },
        },
    }
    return SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False, indent=2)


def _history_before_step(traj: Trajectory, step_index: int) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for prev in traj.steps[:step_index]:
        history.append({"role": "assistant", "content": _format_action(prev)})
        if prev.observation is not None:
            history.append(
                {
                    "role": "user",
                    "content": f"[OUTPUT OF STEP {prev.index}] -> Observation:\n{prev.observation}",
                }
            )
    return history


def _format_action(step: Step) -> str:
    return f"{step.thought.strip()}\n\nCode:\n```py\n{step.code.strip()}\n```".strip()
