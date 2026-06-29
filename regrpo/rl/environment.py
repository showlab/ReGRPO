# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Offline and future online environments for ReGRPO rollouts."""

from __future__ import annotations

import ast
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from regrpo.common.schema import RotRecord, evidence_grounded
from regrpo.common.trajectory import extract_tool_calls, parse_trajectory
from regrpo.rl.core import REFLECTION_COST_SCALE, compute_reward, is_degenerate_group

CODE_BLOCK_RE = re.compile(r"```(?:py|python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class Observation:
    """Normalized tool output."""

    status: str
    payload: str
    meta: dict


class Environment(ABC):
    """Environment interface shared by replay and future online rollouts."""

    @abstractmethod
    def reset(self, state: Any) -> Observation:
        """Reset to a state and return the current observation."""

    @abstractmethod
    def step(self, action: str) -> Observation:
        """Execute or replay one action."""

    @abstractmethod
    def reward(self, traj: Any) -> float:
        """Return a scalar trajectory reward."""

    @abstractmethod
    def success(self, traj: Any) -> bool:
        """Return whether a trajectory succeeded."""


@dataclass
class Candidate:
    """One contrastive group member.

    ``verifier_score`` carries the per-candidate verifier value V used by the
    ``lambda_val * V`` reward term. It defaults to 0.0 and is only populated by
    ``OfflineReplayEnvironment.contrastive_group`` when ``lambda_val > 0``, so a
    ``lambda_val == 0`` run leaves every candidate at 0.0 and reproduces the
    verifier-off reward byte-for-byte.
    """

    code: str
    reflection: str
    success: bool
    label: str
    verifier_score: float = 0.0


def normalized_tool_signature(code: str) -> tuple[str | None, str | None]:
    """Return primary tool and normalized first argument for step equivalence."""

    calls = extract_tool_calls(_code_payload(code))
    if not calls:
        return None, None
    call = calls[0]
    return call.name, _normalize_arg(call.first_arg_repr)


def action_equivalent(action_code: str, reference_code: str) -> bool:
    """Return whether two steps call the same tool with the same first argument."""

    return normalized_tool_signature(action_code) == normalized_tool_signature(reference_code)


def combine_verifier_scores(
    s_a: float,
    s_g: float,
    s_p: float,
    weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
) -> float:
    """Combine clamped verifier subscores into V."""

    scores = (_clamp01(s_a), _clamp01(s_g), _clamp01(s_p))
    return sum(weight * score for weight, score in zip(weights, scores, strict=True))


class OfflineReplayEnvironment(Environment):
    """Deterministic replay environment for contrastive offline ReGRPO groups.

    Candidate success is mapped at the step level: a replayed candidate succeeds
    when its normalized primary tool and first argument match the stored correct
    action, or when it is the explicit correct/reflect member in the constructed
    group. This avoids brittle raw string matching while keeping offline reward
    deterministic.
    """

    def __init__(
        self,
        records: list[RotRecord | dict],
        *,
        lambda_exec: float = 1.0,
        eta: float = 0.1,
        lambda_val: float = 0.0,
    ) -> None:
        self.records = [_as_record(record) for record in records]
        self.lambda_exec = lambda_exec
        self.eta = eta
        self.lambda_val = lambda_val
        self.current_record: RotRecord | None = None
        self.current_step_index = 0
        self.last_group_degenerate = False

    def reset(self, state: Any) -> Observation:
        """Select a record and step for replay."""

        if isinstance(state, tuple):
            record_or_index, step_index = state
            record = self.records[record_or_index] if isinstance(record_or_index, int) else _as_record(record_or_index)
            self.current_step_index = int(step_index)
        else:
            record = self.records[int(state)] if isinstance(state, int) else _as_record(state)
            self.current_step_index = int(record.rot_meta.step_index)
        self.current_record = record
        return Observation(status="empty", payload="", meta={"record_id": record.id})

    def step(self, action: str) -> Observation:
        """Replay stored success observation or return a grounded failure."""

        if self.current_record is None:
            raise RuntimeError("reset must be called before step")
        meta = self.current_record.rot_meta
        if action_equivalent(action, meta.corrected_action):
            traj = parse_trajectory(self.current_record.to_dict())
            if 0 <= self.current_step_index < len(traj.steps):
                payload = traj.steps[self.current_step_index].observation or ""
            else:
                payload = ""
            return Observation(status="ok", payload=payload, meta={"record_id": self.current_record.id})
        return Observation(
            status="tool_error",
            payload=f"Replay mismatch for action at step {self.current_step_index}.",
            meta={"record_id": self.current_record.id, "grounded": True},
        )

    def verifier_subscores(self, candidate: "Candidate") -> tuple[float, float, float]:
        """Return (s_a, s_g, s_p) deterministic verifier subscores for a candidate.

        Subscores are derived from the active record's RoT metadata so V is a
        function of the candidate, not of the policy network:

        - ``s_p`` (plan validity): the candidate's primary tool and first argument
          match the stored ``corrected_action`` signature.
        - ``s_a`` (answer consistency): the candidate's terminal action agrees
          with the group's correct answer (its replay success flag).
        - ``s_g`` (grounding): the candidate carries a reflection whose evidence
          is text-grounded in the stored ``failed_observation`` AND its action is
          plan-valid. There is intentionally NO fallback to ``s_p`` for
          reflection-less candidates: grounding is only credited when an actual
          grounded reflection is present, so V rewards grounded reflection above
          a bare-correct retry instead of restating plan validity.
        """

        if self.current_record is None:
            raise RuntimeError("reset must be called before verifier_subscores")
        meta = self.current_record.rot_meta
        s_p = 1.0 if action_equivalent(candidate.code, meta.corrected_action) else 0.0
        s_a = 1.0 if candidate.success else 0.0
        grounded_reflection = bool(
            candidate.reflection.strip()
            and evidence_grounded(meta.reflection.evidence, meta.failed_observation)
        )
        s_g = 1.0 if (grounded_reflection and s_p > 0.0) else 0.0
        return s_a, s_g, s_p

    def verifier_score(self, candidate: "Candidate") -> float:
        """Return the combined verifier value V in [0, 1] for a candidate."""

        s_a, s_g, s_p = self.verifier_subscores(candidate)
        return combine_verifier_scores(s_a, s_g, s_p)

    def reward(self, traj: Any) -> float:
        """Compute replay reward with the same formula as core.py.

        The reflection cost C is normalized by ``REFLECTION_COST_SCALE`` (per
        100 tokens/words), matching the trainers so a concise successful
        recovery keeps a positive reward gap over an unrecovered failure.
        """

        return compute_reward(
            self.success(traj),
            _reflection_count_for_reward(traj) / REFLECTION_COST_SCALE,
            getattr(traj, "verifier_score", 0.0),
            lambda_exec=self.lambda_exec,
            eta=self.eta,
            lambda_val=self.lambda_val,
        )

    def success(self, traj: Any) -> bool:
        """Return candidate success for replay objects."""

        if isinstance(traj, Candidate):
            return bool(traj.success)
        return bool(getattr(traj, "success", False))

    def contrastive_group(self, record: RotRecord | dict, step_index: int) -> list[Candidate]:
        """Build a non-degenerate offline group for one RoT step."""

        rec = _as_record(record)
        meta = rec.rot_meta
        correct = meta.corrected_action
        failed = meta.failed_action
        reflection = _format_reflection(meta.reflection)
        raw = [
            ("correct", correct, "", True),
            ("failed", failed, "", False),
            ("reflect_correct", correct, reflection, True),
            ("retry_correct", correct, "", False),
            ("corrupted", self._corrupt_action(correct), "", False),
        ]
        group = [
            Candidate(
                code=code,
                reflection=refl,
                success=bool(explicit_success or action_equivalent(code, correct)),
                label=label,
            )
            for label, code, refl, explicit_success in raw
        ]
        # Populate the verifier value V only when the verifier reward is active.
        # When lambda_val == 0 the field stays at its 0.0 default and the reward
        # is byte-identical to the verifier-off path. self.current_record is set
        # so verifier_score can read this record's RoT metadata.
        if self.lambda_val > 0.0:
            self.current_record = rec
            self.current_step_index = int(step_index)
            for candidate in group:
                candidate.verifier_score = self.verifier_score(candidate)
        rewards = [compute_reward(candidate.success, 0) for candidate in group]
        self.last_group_degenerate = is_degenerate_group(rewards)
        if self.last_group_degenerate:
            warnings.warn(
                f"degenerate contrastive group for record={rec.id} step={step_index}",
                RuntimeWarning,
                stacklevel=2,
            )
        return group

    def _corrupt_action(self, code: str) -> str:
        tool, _ = normalized_tool_signature(code)
        replacement = "final_answer" if tool != "final_answer" else "ask_search_agent"
        return f"Thought: corrupted replay action\nCode:\n```py\n{replacement}(\"\")\n```"


class ToolEnvironment(Environment):
    """Future online environment backed by real tool execution.

    The online path will run generated actions against MAT-compatible tools,
    normalize tool outputs into Observation, and use the same reward interface
    as OfflineReplayEnvironment. It is intentionally not implemented in this
    offline dispatch.
    """

    def reset(self, state: Any) -> Observation:
        raise NotImplementedError

    def step(self, action: str) -> Observation:
        raise NotImplementedError

    def reward(self, traj: Any) -> float:
        raise NotImplementedError

    def success(self, traj: Any) -> bool:
        raise NotImplementedError


def _normalize_arg(arg: str | None) -> str | None:
    if arg is None:
        return None
    stripped = arg.strip()
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        value = stripped
    return str(value).strip().lower()


def _code_payload(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text


def _as_record(record: RotRecord | dict) -> RotRecord:
    return record if isinstance(record, RotRecord) else RotRecord.from_dict(record)


def _format_reflection(reflection: Any) -> str:
    error_type = getattr(reflection.error_type, "value", str(reflection.error_type))
    return "\n".join(
        [
            "Reflection:",
            f"- ErrorType: {error_type}",
            f"- Evidence: {reflection.evidence}",
            f"- FixPlan: {reflection.fix_plan}",
            "",
        ]
    )


def _reflection_count_for_reward(traj: Any) -> int:
    if isinstance(traj, Candidate):
        return len(traj.reflection.split())
    return int(getattr(traj, "reflection_count", 0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
