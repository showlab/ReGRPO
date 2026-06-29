# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Verifier-free ReGRPO inference agent."""

from __future__ import annotations

import argparse
from importlib import metadata
from collections.abc import Callable, Sequence
from typing import Any

from regrpo import try_import_mat
from regrpo.common.trajectory import Step, parse_trajectory
from regrpo.inference.trigger import ReflectionGate, normalize_observation
from regrpo.rl.environment import Observation, OfflineReplayEnvironment


def _load_mat_react_base() -> type | None:
    """Return MAT's ReactCodeAgent-compatible class when importable."""

    try:
        transformers_major = int(metadata.version("transformers").split(".", maxsplit=1)[0])
    except (metadata.PackageNotFoundError, ValueError):
        transformers_major = 0
    if transformers_major >= 5:
        return None
    mat = try_import_mat()
    if mat is None:
        return None
    try:
        from tongagent.agents.gaia_agent import ReactCodeGAIAAgent
    except Exception:
        return None
    return ReactCodeGAIAAgent


class _MinimalReactBase:
    """Small ReAct loop skeleton used when MAT cannot import.

    Current CPU-only environments can have transformers versions without
    transformers.agents, which makes MAT's ReactCodeAgent unavailable. This
    fallback keeps memory, a step placeholder, and a simple run loop so the
    verifier-free trigger can be exercised against OfflineReplayEnvironment.
    """

    def __init__(
        self,
        *,
        actions: Sequence[str] | None = None,
        action_logprobs: Sequence[Sequence[float]] | None = None,
        **_: Any,
    ) -> None:
        self.memory: list[dict[str, Any]] = []
        self.actions = list(actions or [])
        self.action_logprobs = [list(item) for item in (action_logprobs or [])]
        self._cursor = 0

    def step(self, *_: Any, **__: Any) -> tuple[str, list[float]]:
        """Return the next queued action and its action-token logprobs."""

        if self._cursor >= len(self.actions):
            raise StopIteration
        action = self.actions[self._cursor]
        logprobs = (
            self.action_logprobs[self._cursor]
            if self._cursor < len(self.action_logprobs)
            else [-0.1]
        )
        self._cursor += 1
        return action, list(logprobs)

    def run(self, env: OfflineReplayEnvironment, max_steps: int) -> list[Observation]:
        """Run queued actions through an environment."""

        observations: list[Observation] = []
        for _ in range(max_steps):
            try:
                action, _ = self.step()
            except StopIteration:
                break
            obs = env.step(action)
            self.memory.append({"role": "assistant", "content": action})
            self.memory.append({"role": "user", "content": obs.payload})
            observations.append(obs)
        return observations


MAT_ReactCodeAgent = _load_mat_react_base()
BASE = MAT_ReactCodeAgent if MAT_ReactCodeAgent is not None else _MinimalReactBase


class ReGRPOAgent(BASE):
    """ReAct-style agent with verifier-free one-shot reflection blocks."""

    def __init__(
        self,
        *args: Any,
        reflection_policy: Callable[[dict[str, Any]], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "memory"):
            self.memory = []
        self.reflection_gate = ReflectionGate()
        self.reflection_policy = reflection_policy or _default_reflection_policy
        self.reflection_log: list[dict[str, Any]] = []

    def run_offline(
        self,
        env: OfflineReplayEnvironment,
        record: Any,
        max_steps: int,
    ) -> dict[str, Any]:
        """Drive one parsed trajectory over OfflineReplayEnvironment."""

        typed_record = getattr(record, "to_dict", lambda: record)()
        traj = parse_trajectory(typed_record)
        env.reset(record)
        transcript: list[dict[str, Any]] = []
        steps = [step for step in traj.steps if not step.is_final][:max_steps]
        for step_number, step in enumerate(steps, start=1):
            action, logprobs = self._offline_action(step, step_number)
            obs = self._offline_observation(env, record, step, action)
            transcript.append(
                {
                    "step_index": step_number,
                    "action": action,
                    "observation": obs.payload,
                    "status": obs.status,
                }
            )
            self.memory.append({"role": "assistant", "content": action})
            self.memory.append({"role": "user", "content": obs.payload})
            if self.reflection_gate.should_reflect(step_number, logprobs, obs):
                decision = dict(self.reflection_gate.last_decision or {})
                reflection = self.reflection_policy(decision)
                corrected_action = self._corrected_action(record, step)
                corrected_obs = self._offline_observation(env, record, step, corrected_action)
                entry = {
                    "step_index": step_number,
                    "reason": decision.get("reason", "none"),
                    "u_i": decision.get("u_i"),
                    "kappa_i": decision.get("kappa_i"),
                    "reflection": reflection,
                }
                self.reflection_log.append(entry)
                transcript.append(
                    {
                        "step_index": step_number,
                        "reflection": reflection,
                        "corrected_action": corrected_action,
                        "corrected_observation": corrected_obs.payload,
                        "corrected_status": corrected_obs.status,
                    }
                )
                self.memory.append({"role": "assistant", "content": reflection + "\n" + corrected_action})
                self.memory.append({"role": "user", "content": corrected_obs.payload})
        return {
            "steps": len(steps),
            "reflections_fired": len(self.reflection_log),
            "reflection_log": list(self.reflection_log),
            "transcript": transcript,
            "base": BASE.__name__,
        }

    def _offline_action(self, step: Step, step_number: int) -> tuple[str, list[float]]:
        if isinstance(self, _MinimalReactBase) and self._cursor < len(self.actions):
            return self.step()
        return _format_action(step), _default_logprobs(step_number)

    def _offline_observation(
        self,
        env: OfflineReplayEnvironment,
        record: Any,
        step: Step,
        action: str,
    ) -> Observation:
        meta = getattr(record, "rot_meta", None)
        if meta is not None and step.index == int(meta.step_index):
            env.current_step_index = step.index
            return normalize_observation(env.step(action))
        return normalize_observation(step.observation)

    def _corrected_action(self, record: Any, step: Step) -> str:
        meta = getattr(record, "rot_meta", None)
        if meta is not None and step.index == int(meta.step_index):
            return str(meta.corrected_action)
        return _format_action(step)


def _format_action(step: Step) -> str:
    return f"{step.thought}\nCode:\n```py\n{step.code}\n```".strip()


def _default_logprobs(step_number: int) -> list[float]:
    return [-0.1]


def _default_reflection_policy(decision: dict[str, Any]) -> str:
    reason = decision.get("reason", "trigger")
    return "\n".join(
        [
            "Reflection:",
            f"- ErrorType: {reason}",
            f"- Evidence: trigger fired from {reason}",
            "- FixPlan: revise the action once using the latest observation.",
        ]
    )


def _synthetic_record() -> Any:
    from regrpo.common.schema import ErrorType, Reflection, RotMeta, RotRecord

    failed = 'Thought: inspect wrong file\nCode:\n```py\ninspect_file_as_text("missing.txt")\n```'
    corrected = 'Thought: inspect correct file\nCode:\n```py\ninspect_file_as_text("report.txt")\n```'
    return RotRecord(
        id="synthetic_rot",
        image={},
        answer="42",
        conversations=[
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "Task: read the report"},
            {
                "role": "assistant",
                "content": 'Thought: search first\nCode:\n```py\nask_search_agent("report location")\n```',
            },
            {"role": "user", "content": "[OUTPUT OF STEP 0] -> Observation:\nreport.txt"},
            {"role": "assistant", "content": failed},
            {"role": "user", "content": "[OUTPUT OF STEP 1] -> Observation:\nanswer: 42"},
            {"role": "assistant", "content": 'Thought: answer\nCode:\n```py\nfinal_answer("42")\n```'},
        ],
        rot_meta=RotMeta(
            src_id="synthetic",
            step_index=1,
            error_type=ErrorType.ARG_INVALID,
            failed_action=failed,
            failed_observation="File not found: missing.txt",
            reflection=Reflection(
                error_type=ErrorType.ARG_INVALID,
                evidence="File not found",
                fix_plan="Use report.txt instead.",
            ),
            corrected_action=corrected,
            final_answer="42",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline ReGRPO inference smoke.")
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    record = _synthetic_record()
    env = OfflineReplayEnvironment([record])
    agent = ReGRPOAgent()
    result = agent.run_offline(env, record, max_steps=args.max_steps)
    print(f"base={result['base']}")
    print(f"steps={result['steps']} reflections_fired={result['reflections_fired']}")
    print(f"reflection_log={result['reflection_log']}")
    for item in result["transcript"]:
        print(f"transcript={item}")


if __name__ == "__main__":
    main()
