# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""RoT record schema and deterministic validation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

MIN_GROUNDED_TOKEN_LEN = 4
MIN_TOKEN_OVERLAP = 2
TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")


def _response_role() -> str:
    return "assistant"


class ErrorType(str, Enum):
    """Perturbation categories used by RoT records."""

    GROUNDING_DRIFT = "GroundingDrift"
    TOOL_MISMATCH = "ToolMismatch"
    ARG_INVALID = "ArgInvalid"
    INFO_INSUFFICIENT = "InfoInsufficient"
    # NonCommit targets the FINAL step: the trajectory reached the answer but the
    # failed action rambles in prose without emitting a clean final_answer() call.
    # The corrected action is the trajectory's real, concise final_answer(...) step.
    NON_COMMIT = "NonCommit"
    # CodeError targets a buggy code cell whose Python execution raised a real
    # traceback (e.g. NameError, TypeError, 'has no attribute'). The agent
    # otherwise repeats the broken code until max-execution. The failed
    # observation IS the traceback, the reflection quotes it, and the corrected
    # action is the fixed code cell that resolves to the ground-truth answer.
    # Additive and gated: existing generation (perturb.FEASIBILITY) never emits
    # CodeError, so the base RoT pipeline is byte-for-byte unaffected.
    CODE_ERROR = "CodeError"


@dataclass
class Reflection:
    """Structured reflection payload."""

    error_type: ErrorType
    evidence: str
    fix_plan: str


@dataclass
class RotMeta:
    """Structured metadata for one RoT example."""

    src_id: str
    step_index: int
    error_type: ErrorType
    failed_action: str
    failed_observation: str
    reflection: Reflection
    corrected_action: str
    final_answer: str


@dataclass
class RotRecord:
    """Hybrid RoT record with inline conversations and metadata."""

    id: str
    image: Any
    answer: Any
    conversations: list[dict]
    rot_meta: RotMeta

    def to_dict(self) -> dict:
        """Serialize dataclasses and enum values to plain dicts."""

        return {
            "id": self.id,
            "image": self.image,
            "answer": self.answer,
            "conversations": self.conversations,
            "rot_meta": {
                "src_id": self.rot_meta.src_id,
                "step_index": self.rot_meta.step_index,
                "error_type": self.rot_meta.error_type.value,
                "failed_action": self.rot_meta.failed_action,
                "failed_observation": self.rot_meta.failed_observation,
                "reflection": {
                    "error_type": self.rot_meta.reflection.error_type.value,
                    "evidence": self.rot_meta.reflection.evidence,
                    "fix_plan": self.rot_meta.reflection.fix_plan,
                },
                "corrected_action": self.rot_meta.corrected_action,
                "final_answer": self.rot_meta.final_answer,
            },
        }

    @classmethod
    def from_dict(cls, obj: dict) -> "RotRecord":
        """Build a typed record from a plain dict."""

        meta = obj["rot_meta"]
        refl = meta["reflection"]
        reflection = Reflection(
            error_type=ErrorType(refl["error_type"]),
            evidence=refl["evidence"],
            fix_plan=refl["fix_plan"],
        )
        rot_meta = RotMeta(
            src_id=meta["src_id"],
            step_index=int(meta["step_index"]),
            error_type=ErrorType(meta["error_type"]),
            failed_action=meta["failed_action"],
            failed_observation=meta["failed_observation"],
            reflection=reflection,
            corrected_action=meta["corrected_action"],
            final_answer=meta["final_answer"],
        )
        return cls(
            id=obj["id"],
            image=obj.get("image"),
            answer=obj.get("answer"),
            conversations=list(obj["conversations"]),
            rot_meta=rot_meta,
        )


def _as_dict(rec: dict | RotRecord) -> dict:
    return rec.to_dict() if isinstance(rec, RotRecord) else rec


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if len(token) >= 2}


def evidence_grounded(evidence: str, failed_observation: str) -> bool:
    """Return whether evidence is text-grounded in the failed observation.

    The heuristic accepts either a meaningful literal token from the
    observation appearing in the evidence, or at least MIN_TOKEN_OVERLAP shared
    normalized tokens. This rejects generic claims that do not cite the
    observed failure text.
    """

    if not evidence or not failed_observation:
        return False
    evidence_norm = evidence.lower()
    obs_tokens = _tokens(failed_observation)
    for token in obs_tokens:
        if len(token) >= MIN_GROUNDED_TOKEN_LEN and token in evidence_norm:
            return True
    evidence_tokens = _tokens(evidence)
    return len(obs_tokens & evidence_tokens) >= MIN_TOKEN_OVERLAP


def has_label_leak(conversations: list[dict], corrected_action: str) -> bool:
    """Return whether corrected action text appears before the target turn."""

    if not corrected_action:
        return False
    final_index = _final_response_index(conversations)
    if final_index <= 0:
        return False
    needle = corrected_action.strip()
    if not needle:
        return False
    for turn in conversations[:final_index]:
        if needle in str(turn.get("content", "")):
            return True
    return False


def _final_response_index(conversations: list[dict]) -> int:
    for index, turn in enumerate(conversations):
        if turn.get("role") == _response_role() and str(turn.get("content", "")).startswith(
            "Reflection:\n"
        ):
            return index
    for index in range(len(conversations) - 1, -1, -1):
        if conversations[index].get("role") == _response_role():
            return index
    return -1


def _roles_are_sane(conversations: list[dict]) -> bool:
    if not conversations:
        return False
    roles = [turn.get("role") for turn in conversations]
    if roles[0] != "system":
        return False
    expected = "user"
    for role in roles[1:]:
        if role != expected:
            return False
        expected = _response_role() if expected == "user" else "user"
    return True


def validate_rot_record(rec: dict | RotRecord) -> list[str]:
    """Return human-readable validation problems, or an empty list."""

    obj = _as_dict(rec)
    problems: list[str] = []
    for key in ("id", "image", "answer", "conversations", "rot_meta"):
        if key not in obj:
            problems.append(f"missing required key: {key}")
    if problems:
        return problems

    meta = obj.get("rot_meta") or {}
    for key in (
        "src_id",
        "step_index",
        "error_type",
        "failed_action",
        "failed_observation",
        "reflection",
        "corrected_action",
        "final_answer",
    ):
        if key not in meta:
            problems.append(f"missing rot_meta key: {key}")
    if problems:
        return problems

    try:
        ErrorType(meta["error_type"])
    except ValueError:
        problems.append(f"invalid rot_meta.error_type: {meta.get('error_type')}")

    reflection = meta.get("reflection") or {}
    for key in ("error_type", "evidence", "fix_plan"):
        if key not in reflection:
            problems.append(f"missing reflection key: {key}")
    if "error_type" in reflection:
        try:
            ErrorType(reflection["error_type"])
        except ValueError:
            problems.append(f"invalid reflection.error_type: {reflection.get('error_type')}")

    conversations = obj.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        problems.append("conversations must be a non-empty list")
    elif not _roles_are_sane(conversations):
        problems.append("conversation roles do not alternate sanely")

    if reflection and not evidence_grounded(
        str(reflection.get("evidence", "")),
        str(meta.get("failed_observation", "")),
    ):
        problems.append("reflection.evidence is not grounded in failed_observation")

    if isinstance(conversations, list):
        # Only the corrected action a* (the reflection-turn SFT target) must not
        # leak before the target turn. The trajectory's final_answer is NOT the
        # SFT target — it legitimately appears in earlier observations (e.g. a
        # search result that already contains the answer), so it is not checked.
        if has_label_leak(conversations, str(meta.get("corrected_action", ""))):
            problems.append("corrected_action leaks before target turn")
    return problems
