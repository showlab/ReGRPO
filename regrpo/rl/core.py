# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Pure ReGRPO reward, advantage, logprob, and KL helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    """One rollout candidate with explicit action/reflection factorization."""

    action_logprobs: "torch.Tensor"
    reflection_logprobs: "torch.Tensor"
    ref_action_logprobs: "torch.Tensor"
    ref_reflection_logprobs: "torch.Tensor"
    success: bool
    verifier_score: float = 0.0


# Reflection cost is measured per this many tokens. The paper defines C(tau) as
# "proportional to the token length of z_i"; with lambda_exec = 1 the penalty
# must stay a small fraction of the unit success reward so that a successful,
# concise recovery keeps a positive reward gap (Delta R > 0) over an unrecovered
# failure. Scoring cost per 100 tokens makes eta = 0.1 a ~3% penalty for a
# typical 30-token reflection instead of a 3x over-penalty on raw token counts.
REFLECTION_COST_SCALE = 100.0


def reflection_token_count(r: Rollout) -> int:
    """Return the number of reflection tokens in a rollout."""

    return int(r.reflection_logprobs.numel())


def reflection_cost(r: Rollout, scale: float = REFLECTION_COST_SCALE) -> float:
    """Return the reflection-length cost C(tau), normalized by ``scale`` tokens.

    C is 0 for a one-shot success (no reflection tokens) and grows linearly with
    reflection length, matching the paper's reflection-cost penalty.
    """

    return reflection_token_count(r) / float(scale)


def scored_token_count(r: Rollout) -> int:
    """Return the number of action plus reflection tokens scored by the loss."""

    return int(r.action_logprobs.numel() + r.reflection_logprobs.numel())


def compute_reward(
    success: bool,
    reflection_count: int,
    verifier_score: float = 0.0,
    *,
    lambda_exec: float = 1.0,
    eta: float = 0.1,
    lambda_val: float = 0.0,
) -> float:
    """Compute R = lambda_exec * success - eta * C + lambda_val * V."""

    return (
        lambda_exec * float(success)
        - eta * float(reflection_count)
        + lambda_val * float(verifier_score)
    )


def group_advantage(
    rewards: Sequence[float],
    *,
    normalize: bool = False,
    clip_range: float | None = None,
) -> list[float]:
    """Mean-center rewards with optional group normalization and clipping."""

    if not rewards:
        return []
    mean_reward = sum(float(reward) for reward in rewards) / len(rewards)
    advantages = [float(reward) - mean_reward for reward in rewards]
    if normalize:
        std = math.sqrt(group_reward_variance(rewards))
        advantages = [advantage / (std + 1e-8) for advantage in advantages]
    if clip_range is not None:
        limit = float(clip_range)
        advantages = [max(-limit, min(limit, advantage)) for advantage in advantages]
    return advantages


def group_advantage_two_stream(
    r_outcome: Sequence[float],
    v: Sequence[float],
    alpha: float,
    *,
    normalize: bool = False,
    clip_range: float | None = None,
) -> list[float]:
    """Add independently normalized outcome and verifier advantages.

    The verifier value V is normalized by its OWN within-group mean/sigma, so
    it is scale-free and immune to success-variance washout from the outcome
    stream. ``clip_range`` is applied per stream, matching the bounded
    advantage-clip semantics of the stability recipe.
    """

    if len(r_outcome) != len(v):
        raise ValueError("r_outcome and v must have the same length")
    if not r_outcome:
        return []
    a_outcome = group_advantage(r_outcome, normalize=normalize, clip_range=clip_range)
    a_v = group_advantage(v, normalize=normalize, clip_range=clip_range)
    return [o + float(alpha) * w for o, w in zip(a_outcome, a_v, strict=True)]


def group_advantage_loo(rewards: Sequence[float]) -> list[float]:
    """Leave-one-out group baseline: A_i = R_i - mean(R_j for j != i)."""

    if not rewards:
        return []
    if len(rewards) < 2:
        raise ValueError("leave-one-out advantage requires at least two rewards")
    reward_values = [float(reward) for reward in rewards]
    total = sum(reward_values)
    denom = len(reward_values) - 1
    return [reward - ((total - reward) / denom) for reward in reward_values]


def sequence_logprob(r: Rollout, *, length_normalize: bool = False) -> torch.Tensor:
    """Return log pi over action plus reflection tokens."""

    logprob = r.action_logprobs.sum() + r.reflection_logprobs.sum()
    if not length_normalize:
        return logprob
    return logprob / max(scored_token_count(r), 1)


def kl_to_ref(
    r: Rollout,
    estimator: str = "k1",
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Estimate sequence-level KL to the frozen reference policy.

    The default ``k1`` estimator is sum(log pi_theta - log pi_ref). The
    low-variance non-negative ``k3`` estimator, exp(delta) - delta - 1, is
    available as ``kl_to_ref(r, estimator="k3")``. When length-normalized, the
    same scored action/reflection span is averaged per token.
    """

    policy = torch.cat((r.action_logprobs, r.reflection_logprobs))
    ref = torch.cat((r.ref_action_logprobs, r.ref_reflection_logprobs))
    delta = policy - ref
    if estimator == "k1":
        kl = delta.sum()
    elif estimator == "k3":
        kl = (torch.exp(delta) - delta - 1.0).sum()
    else:
        raise ValueError(f"unknown KL estimator: {estimator}")
    if not length_normalize:
        return kl
    return kl / max(scored_token_count(r), 1)


def regrpo_loss(
    rollouts: Sequence[Rollout],
    advantages: Sequence[float],
    beta: float,
    *,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Compute the differentiable ReGRPO objective for one contrastive group."""

    if len(rollouts) != len(advantages):
        raise ValueError("rollouts and advantages must have the same length")
    if not rollouts:
        raise ValueError("at least one rollout is required")
    terms = []
    kls = []
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        adv = torch.as_tensor(
            float(advantage),
            dtype=rollout.action_logprobs.dtype,
            device=rollout.action_logprobs.device,
        ).detach()
        terms.append(adv * sequence_logprob(rollout, length_normalize=length_normalize))
        kls.append(kl_to_ref(rollout, length_normalize=length_normalize))
    policy_loss = -torch.stack(terms).mean()
    kl_loss = torch.stack(kls).mean()
    return policy_loss + float(beta) * kl_loss


def group_reward_variance(rewards: Sequence[float]) -> float:
    """Return population variance for a reward group."""

    if not rewards:
        return 0.0
    mean_reward = sum(float(reward) for reward in rewards) / len(rewards)
    return sum((float(reward) - mean_reward) ** 2 for reward in rewards) / len(rewards)


def is_degenerate_group(rewards: Sequence[float], tol: float = 1e-8) -> bool:
    """Return whether reward variance is too small for group advantages."""

    return group_reward_variance(rewards) <= tol
