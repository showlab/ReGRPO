# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Minimal local-model ReGRPO training loop."""

from __future__ import annotations

import argparse
import itertools
import os
import random
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from regrpo.common.io import read_json
from regrpo.common.schema import RotRecord
from regrpo.rl.core import (
    Rollout,
    compute_reward,
    group_advantage,
    group_reward_variance,
    is_degenerate_group,
    reflection_cost,
    reflection_token_count,
    regrpo_loss,
)
from regrpo.rl.environment import Candidate, OfflineReplayEnvironment


class MinimalReGRPOTrainer:
    """Small CPU-runnable trainer using local AutoModelForCausalLM + torch."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        lambda_exec: float = 1.0,
        eta: float = 0.1,
        lambda_val: float = 0.0,
        beta: float = 0.04,
        lr: float = 1e-6,
        use_lora: bool = True,
        device: str = "cpu",
        max_len: int = 1024,
        output_dir: str | None = None,
        init_adapter: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.init_adapter = init_adapter
        self.lambda_exec = lambda_exec
        self.eta = eta
        self.lambda_val = lambda_val
        self.beta = beta
        self.max_len = int(max_len)
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        init_adapter_path = init_adapter.strip() if init_adapter else ""
        if init_adapter_path:
            # Continue RL from the Stage-1 RoT-SFT LoRA adapter (paper warm start).
            from peft import PeftModel

            self.policy = PeftModel.from_pretrained(
                self.policy, init_adapter_path, is_trainable=True
            )
        elif use_lora:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
            self.policy = get_peft_model(self.policy, config)
        self.policy.to(self.device)
        self.policy.train()
        self.ref = AutoModelForCausalLM.from_pretrained(model_name)
        if init_adapter_path:
            from peft import PeftModel

            self.ref = PeftModel.from_pretrained(
                self.ref, init_adapter_path, is_trainable=False
            )
        self.ref.to(self.device)
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.env: OfflineReplayEnvironment | None = None

    def _candidate_logprobs(
        self,
        context_messages: list[dict],
        candidate: Candidate,
    ) -> Rollout:
        """Return action/reflection token logprobs using chat-prefix spans."""

        prefix = self.tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reflection = candidate.reflection
        action = candidate.code
        prefix_len = len(self.tokenizer(prefix, add_special_tokens=False).input_ids)
        reflection_end = len(
            self.tokenizer(prefix + reflection, add_special_tokens=False).input_ids
        )
        action_end = len(
            self.tokenizer(prefix + reflection + action, add_special_tokens=False).input_ids
        )
        full_text = prefix + reflection + action + "<|im_end|>\n"
        input_ids = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids
        # Long MAT system prompts make full-sequence CPU forwards impractical for a
        # smoke. Left-truncate the non-scored prefix while keeping the reflection and
        # action spans (which sit at the tail) intact, then shift span boundaries.
        offset = 0
        total = int(input_ids.shape[1])
        if self.max_len and total > self.max_len:
            offset = total - self.max_len
            if offset >= prefix_len:
                # The first reflection token's logprob needs the token at
                # prefix_len-1 in-window, so offset must stay below prefix_len;
                # otherwise (z, a*) would be silently scored only in part.
                raise ValueError(
                    "max_len truncated the scored reflection/action span; "
                    f"increase max_len above {total - prefix_len + 1}"
                )
            input_ids = input_ids[:, offset:]
        input_ids = input_ids.to(self.device)
        policy_logprobs = self._token_logprobs(self.policy, input_ids)
        with torch.no_grad():
            ref_logprobs = self._token_logprobs(self.ref, input_ids)

        reflection_slice = _span_to_logprob_slice(prefix_len - offset, reflection_end - offset)
        action_slice = _span_to_logprob_slice(reflection_end - offset, action_end - offset)
        return Rollout(
            action_logprobs=policy_logprobs[action_slice],
            reflection_logprobs=policy_logprobs[reflection_slice],
            ref_action_logprobs=ref_logprobs[action_slice],
            ref_reflection_logprobs=ref_logprobs[reflection_slice],
            success=candidate.success,
            verifier_score=getattr(candidate, "verifier_score", 0.0),
        )

    def train_step(self, record: RotRecord | dict, step_index: int) -> dict:
        """Run one contrastive ReGRPO update, or skip a degenerate group."""

        rec = record if isinstance(record, RotRecord) else RotRecord.from_dict(record)
        if self.env is None:
            self.env = OfflineReplayEnvironment(
                [rec],
                lambda_exec=self.lambda_exec,
                eta=self.eta,
                lambda_val=self.lambda_val,
            )
        group = self.env.contrastive_group(rec, step_index)
        context_messages = _context_messages(rec, step_index)
        rollouts = [self._candidate_logprobs(context_messages, candidate) for candidate in group]
        rewards = [
            compute_reward(
                rollout.success,
                reflection_cost(rollout),
                rollout.verifier_score,
                lambda_exec=self.lambda_exec,
                eta=self.eta,
                lambda_val=self.lambda_val,
            )
            for rollout in rollouts
        ]
        if is_degenerate_group(rewards):
            warnings.warn("skipping degenerate ReGRPO group", RuntimeWarning, stacklevel=2)
            return {"skipped": "degenerate_group", "reward_var": group_reward_variance(rewards)}

        advantages = group_advantage(rewards)
        loss = regrpo_loss(rollouts, advantages, self.beta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(sum(rewards) / len(rewards)),
            "reward_var": group_reward_variance(rewards),
            "n_reflection_tokens": int(sum(reflection_token_count(rollout) for rollout in rollouts)),
        }

    def fit(self, records: list[RotRecord | dict], max_steps: int) -> list[dict]:
        """Train for a small number of steps over RoT records."""

        typed = [record if isinstance(record, RotRecord) else RotRecord.from_dict(record) for record in records]
        self.env = OfflineReplayEnvironment(
            typed,
            lambda_exec=self.lambda_exec,
            eta=self.eta,
            lambda_val=self.lambda_val,
        )
        metrics = []
        for step, record in zip(range(max_steps), itertools.cycle(typed), strict=False):
            result = self.train_step(record, int(record.rot_meta.step_index))
            result["step"] = step
            print(result)
            metrics.append(result)
        self.save()
        return metrics

    def save(self) -> None:
        """Persist the trained policy adapter and tokenizer."""

        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"saved ReGRPO adapter to {self.output_dir}")

    def _token_logprobs(self, model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.data:
        config["data"] = args.data
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    random.seed(int(config.get("seed", 13)))
    torch.manual_seed(int(config.get("seed", 13)))
    records = read_json(config["data"])
    trainer = MinimalReGRPOTrainer(
        model_name=config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        lambda_exec=float(config.get("lambda_exec", 1.0)),
        eta=float(config.get("eta", 0.1)),
        lambda_val=float(config.get("lambda_val", 0.0)),
        beta=float(config.get("beta", 0.04)),
        lr=float(config.get("lr", 1e-6)),
        use_lora=bool(config.get("use_lora", True)),
        device=str(config.get("device", "cpu")),
        max_len=int(config.get("max_len", 1024)),
        output_dir=config.get("output_dir"),
        init_adapter=config.get("init_adapter"),
    )
    trainer.fit(records, int(config.get("max_steps", 2)))


def _context_messages(record: RotRecord, step_index: int) -> list[dict]:
    count = 0
    for index, turn in enumerate(record.conversations):
        if turn.get("role") != "assistant":
            continue
        if count == step_index:
            return [dict(item) for item in record.conversations[:index]]
        count += 1
    return [dict(item) for item in record.conversations[:-1]]


def _span_to_logprob_slice(start_token: int, end_token: int) -> slice:
    if end_token <= start_token:
        return slice(0, 0)
    return slice(max(start_token - 1, 0), max(end_token - 1, 0))


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
