# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Verifier-free inference trigger from PLAN section 6.1."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from regrpo.rl.environment import Observation

ERROR_MARKERS = (
    "file not found",
    "error:",
    "traceback",
    "exception",
    "http error",
    "http 400",
    "http 401",
    "http 403",
    "http 404",
    "http 408",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "timed out",
    "timeout",
)
EMPTY_MARKERS = {"", "[]", "{}", "none", "null"}


def confidence(action_logprobs: Sequence[float]) -> float:
    """Return u_i = exp(mean(action token logprobs)).

    The confidence span is the action token span only: the Code/action segment
    for a_i, not the full assistant text and not the thought tokens. The caller
    is responsible for slicing model logprobs to that action-token span before
    calling this pure helper. Empty input returns 0.0 because no action-token
    evidence is available.
    """

    if not action_logprobs:
        return 0.0
    return math.exp(sum(action_logprobs) / len(action_logprobs))


def adaptive_threshold(prev_us: Sequence[float]) -> float:
    """Return kappa_i = mean(previous u_j), or -inf when no prior u exists."""

    if not prev_us:
        return -math.inf
    return sum(prev_us) / len(prev_us)


def normalize_observation(raw: Any) -> Observation:
    """Normalize raw tool output into Observation(status, payload, meta).

    Status is one of "ok", "tool_error", or "empty". Exceptions and common
    error strings map to tool_error. Empty payloads, whitespace, "[]", "{}",
    and "None" map to empty. Existing Observation values are normalized to the
    same status set and otherwise preserved.
    """

    if isinstance(raw, Observation):
        # status stores the backend code; anything that is not a known clean
        # status ("ok"/"empty") is treated as a tool error so backend error
        # codes (e.g. "http_500") correctly fire ToolError.
        status = raw.status if raw.status in {"ok", "tool_error", "empty"} else "tool_error"
        return Observation(status=status, payload=str(raw.payload), meta=dict(raw.meta))
    if isinstance(raw, BaseException):
        return Observation(
            status="tool_error",
            payload=f"{raw.__class__.__name__}: {raw}",
            meta={"raw_type": raw.__class__.__name__},
        )
    payload = _payload_text(raw)
    stripped = payload.strip()
    lowered = stripped.lower()
    if lowered in EMPTY_MARKERS:
        return Observation(status="empty", payload=payload, meta={"raw_type": type(raw).__name__})
    if any(marker in lowered for marker in ERROR_MARKERS):
        return Observation(status="tool_error", payload=payload, meta={"raw_type": type(raw).__name__})
    return Observation(status="ok", payload=payload, meta={"raw_type": type(raw).__name__})


def tool_error(obs: Observation) -> bool:
    """Return whether the normalized observation is a tool error."""

    return obs.status == "tool_error"


def empty_obs(obs: Observation) -> bool:
    """Return whether the normalized observation is empty."""

    return obs.status == "empty"


def gate(step_index: int, u_i: float, kappa_i: float, obs: Observation) -> bool:
    """Return g_i for a 1-based inference step index.

    The confidence clause follows the paper's i > 1 convention. At
    step_index=1, low confidence cannot trigger reflection by itself.
    """

    return tool_error(obs) or empty_obs(obs) or (step_index > 1 and u_i < kappa_i)


class ReflectionGate:
    """Stateful trigger wrapper with one reflection block allowed per step."""

    def __init__(self) -> None:
        self.prev_us: list[float] = []
        self.step_us: dict[int, float] = {}
        self.local_reflection_counts: dict[int, int] = defaultdict(int)
        self.last_decision: dict[str, Any] | None = None

    def should_reflect(
        self,
        step_index: int,
        action_logprobs: Sequence[float],
        obs: Observation,
    ) -> bool:
        """Return whether to reflect after this step and update trigger state."""

        u_i = confidence(action_logprobs)
        if step_index in self.step_us:
            prior_us = [u for step, u in sorted(self.step_us.items()) if step < step_index]
        else:
            prior_us = list(self.prev_us)
        kappa_i = adaptive_threshold(prior_us)
        raw_decision = gate(step_index, u_i, kappa_i, obs)
        allowed = self.local_reflection_counts[step_index] < 1
        decision = raw_decision and allowed
        if step_index not in self.step_us:
            self.step_us[step_index] = u_i
            self.prev_us.append(u_i)
        if decision:
            self.local_reflection_counts[step_index] += 1
        self.last_decision = {
            "step_index": step_index,
            "u_i": u_i,
            "kappa_i": kappa_i,
            "obs_status": obs.status,
            "raw_gate": raw_decision,
            "allowed": allowed,
            "decision": decision,
            "reason": _reason(step_index, u_i, kappa_i, obs),
        }
        return decision


def _payload_text(raw: Any) -> str:
    if raw is None:
        return "None"
    if isinstance(raw, (list, tuple, set, dict)) and not raw:
        return "[]" if not isinstance(raw, dict) else "{}"
    return str(raw)


def _reason(step_index: int, u_i: float, kappa_i: float, obs: Observation) -> str:
    if tool_error(obs):
        return "tool_error"
    if empty_obs(obs):
        return "empty"
    if step_index > 1 and u_i < kappa_i:
        return "low_confidence"
    return "none"
