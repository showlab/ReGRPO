# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Qwen2-VL ReGRPO trainer for vision Stage-2.

This module requires the MAT/Qwen-VL GPU environment at runtime: a
transformers build with Qwen2-VL support, qwen_vl_utils, peft, and torch. It is
not run in CI. Imports for those heavy dependencies are intentionally lazy so
``import regrpo.rl.trainer_qwen_vl`` stays hermetic in the repo test env.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import warnings
from pathlib import Path
from typing import Any

from regrpo.common.io import read_json
from regrpo.common.schema import RotRecord


_SKIPPED_MISSING_IMAGES = 0
_MISSING_IMAGE_WARNING_EMITTED = False
DEFAULT_IMAGE_MIN_PIXELS = 4 * 28 * 28
DEFAULT_IMAGE_MAX_PIXELS = 512 * 28 * 28


class QwenVLReGRPOTrainer:
    """ReGRPO trainer that scores candidate tails with Qwen2-VL inputs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        lambda_exec: float = 1.0,
        eta: float = 0.1,
        lambda_val: float = 0.0,
        two_stream_advantage: bool = False,
        verifier_alpha: float = 1.0,
        beta: float = 0.04,
        lr: float = 5e-6,
        use_lora: bool = True,
        device: str = "cuda",
        max_len: int = 4096,
        output_dir: str | None = None,
        init_adapter: str | None = None,
        gradient_checkpointing: bool = True,
        length_normalize: bool = False,
        advantage_normalize: bool = False,
        advantage_clip: float | None = None,
        advantage_estimator: str = "group_mean",
        grad_clip_norm: float | None = None,
        warmup_steps: int = 0,
        image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
        image_max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
        freeze_modules_to_save: bool = False,
    ) -> None:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.model_name = model_name
        self.init_adapter = init_adapter
        self.lambda_exec = lambda_exec
        self.eta = eta
        self.lambda_val = lambda_val
        self.two_stream_advantage = bool(two_stream_advantage)
        self.verifier_alpha = float(verifier_alpha)
        self.beta = beta
        self.max_len = int(max_len)
        self.output_dir = output_dir
        self.gradient_checkpointing = gradient_checkpointing
        self.length_normalize = length_normalize
        self.advantage_normalize = advantage_normalize
        self.advantage_clip = advantage_clip
        self.advantage_estimator = _validate_advantage_estimator(advantage_estimator)
        self.grad_clip_norm = grad_clip_norm
        self.warmup_steps = int(warmup_steps)
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.freeze_modules_to_save = bool(freeze_modules_to_save)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.policy = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        init_adapter_path = init_adapter.strip() if init_adapter else ""
        if init_adapter_path:
            self.policy = PeftModel.from_pretrained(
                self.policy,
                init_adapter_path,
                is_trainable=True,
            )
            if self.freeze_modules_to_save:
                _freeze_modules_to_save(self.policy)
        elif use_lora:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.policy = get_peft_model(self.policy, config)
        self.policy.to(self.device)
        if self.gradient_checkpointing:
            _enable_policy_gradient_checkpointing(self.policy)
        self.policy.train()
        self.ref = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        if init_adapter_path:
            self.ref = PeftModel.from_pretrained(
                self.ref,
                init_adapter_path,
                is_trainable=False,
            )
        self.ref.to(self.device)
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.policy.parameters() if param.requires_grad],
            lr=lr,
        )
        self._target_lr = float(lr)
        self._optimizer_step_count = 0
        self.env: Any | None = None
        if self._stability_features_enabled():
            print(
                "ReGRPO stability config: "
                f"length_normalize={self.length_normalize}, "
                f"advantage_estimator={self.advantage_estimator}, "
                f"advantage_normalize={self.advantage_normalize}, "
                f"advantage_clip={self.advantage_clip}, "
                f"grad_clip_norm={self.grad_clip_norm}, "
                f"warmup_steps={self.warmup_steps}"
            )
            if self.two_stream_advantage:
                print(
                    "ReGRPO two-stream advantage: "
                    f"alpha={self.verifier_alpha}, lambda_val={self.lambda_val}"
                )

    def _candidate_logprobs(
        self,
        record: RotRecord | dict,
        context_messages: list[dict],
        candidate: Any,
    ) -> Any:
        """Return action/reflection logprobs from a multimodal Qwen2-VL forward."""

        import torch

        rec = record if isinstance(record, RotRecord) else RotRecord.from_dict(record)
        messages = _qwen_vl_messages(
            rec,
            context_messages,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
        )
        prefix_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        reflection = candidate.reflection
        action = candidate.code
        full_messages = messages + [{"role": "assistant", "content": reflection + action}]
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_len, reflection_end, action_end, full_text_len = _candidate_token_boundaries(
            self.processor,
            prefix_text,
            reflection,
            action,
            full_text,
        )
        reflection_count = reflection_end - prefix_len
        action_count = action_end - reflection_end
        end_token_count = max(full_text_len - action_end, 0)
        inputs = self._prepare_inputs(full_text, messages)
        input_ids = inputs["input_ids"]
        total = int(input_ids.shape[1])
        prefix_len, reflection_end, action_end = _tail_scored_token_bounds(
            total,
            reflection_count,
            action_count,
            end_token_count,
        )
        offset = 0
        if self.max_len and total > self.max_len:
            offset = total - self.max_len
            if offset >= prefix_len:
                # Guard the WHOLE scored span: the first reflection token's
                # logprob needs the token at prefix_len-1 in-window, so offset
                # must stay below prefix_len or (z, a*) is scored only in part.
                raise ValueError(
                    "max_len truncated the scored reflection/action span; "
                    f"increase max_len above {total - prefix_len + 1}"
                )
            input_ids = input_ids[:, offset:]
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, offset:]
        inputs = _move_inputs(inputs, self.device)
        policy_logprobs = self._token_logprobs(self.policy, inputs)
        with torch.no_grad():
            ref_logprobs = self._token_logprobs(self.ref, inputs)

        reflection_slice, action_slice = _tail_span_slices(
            prefix_len,
            reflection_end,
            action_end,
            offset=offset,
        )
        core = _rl_core()
        return core.Rollout(
            action_logprobs=policy_logprobs[action_slice],
            reflection_logprobs=policy_logprobs[reflection_slice],
            ref_action_logprobs=ref_logprobs[action_slice],
            ref_reflection_logprobs=ref_logprobs[reflection_slice],
            success=candidate.success,
            verifier_score=getattr(candidate, "verifier_score", 0.0),
        )

    def train_step(self, record: RotRecord | dict, step_index: int) -> dict:
        """Run one contrastive ReGRPO update, or skip a degenerate group."""

        import torch

        rec = record if isinstance(record, RotRecord) else RotRecord.from_dict(record)
        core = _rl_core()
        environment = _rl_environment()
        if self.env is None:
            self.env = environment.OfflineReplayEnvironment(
                [rec],
                lambda_exec=self.lambda_exec,
                eta=self.eta,
                lambda_val=self.lambda_val,
            )
        group = self.env.contrastive_group(rec, step_index)
        context_messages = _context_messages(rec, step_index)
        rollouts = [self._candidate_logprobs(rec, context_messages, candidate) for candidate in group]
        rewards = [
            core.compute_reward(
                rollout.success,
                core.reflection_cost(rollout),
                rollout.verifier_score,
                lambda_exec=self.lambda_exec,
                eta=self.eta,
                lambda_val=self.lambda_val,
            )
            for rollout in rollouts
        ]
        if core.is_degenerate_group(rewards):
            warnings.warn("skipping degenerate ReGRPO group", RuntimeWarning, stacklevel=2)
            return {"skipped": "degenerate_group", "reward_var": core.group_reward_variance(rewards)}

        if self.advantage_estimator == "rloo":
            advantages = core.group_advantage_loo(rewards)
            if self.advantage_normalize:
                std = core.group_reward_variance(rewards) ** 0.5
                advantages = [advantage / (std + 1e-8) for advantage in advantages]
            if self.advantage_clip is not None:
                advantages = [
                    max(-float(self.advantage_clip), min(float(self.advantage_clip), advantage))
                    for advantage in advantages
                ]
        else:
            if self.two_stream_advantage:
                r_outcome_list = [
                    core.compute_reward(
                        rollout.success,
                        core.reflection_cost(rollout),
                        0.0,
                        lambda_exec=self.lambda_exec,
                        eta=self.eta,
                        lambda_val=0.0,
                    )
                    for rollout in rollouts
                ]
                v_list = [self.lambda_val * float(rollout.verifier_score) for rollout in rollouts]
                advantages = core.group_advantage_two_stream(
                    r_outcome_list,
                    v_list,
                    self.verifier_alpha,
                    normalize=self.advantage_normalize,
                    clip_range=self.advantage_clip,
                )
            else:
                advantages = core.group_advantage(
                    rewards,
                    normalize=self.advantage_normalize,
                    clip_range=self.advantage_clip,
                )
        loss = core.regrpo_loss(
            rollouts,
            advantages,
            self.beta,
            length_normalize=self.length_normalize,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=float(self.grad_clip_norm))
        self._apply_lr_warmup()
        self.optimizer.step()
        self._optimizer_step_count += 1
        return {
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(sum(rewards) / len(rewards)),
            "reward_var": core.group_reward_variance(rewards),
            "n_reflection_tokens": int(sum(core.reflection_token_count(rollout) for rollout in rollouts)),
        }

    def _apply_lr_warmup(self) -> None:
        """Linearly ramp optimizer LR during the first configured optimizer steps."""

        if self.warmup_steps <= 0:
            return
        scale = min(float(self._optimizer_step_count + 1) / float(self.warmup_steps), 1.0)
        for group in self.optimizer.param_groups:
            group["lr"] = self._target_lr * scale

    def _stability_features_enabled(self) -> bool:
        """Return whether any default-off stability feature is active."""

        return (
            self.length_normalize
            or self.advantage_normalize
            or self.advantage_clip is not None
            or self.advantage_estimator != "group_mean"
            or self.grad_clip_norm is not None
            or self.warmup_steps > 0
            or self.two_stream_advantage
        )

    def fit(self, records: list[RotRecord | dict], max_steps: int) -> list[dict]:
        """Train over RoT records using offline contrastive groups."""

        typed = [record if isinstance(record, RotRecord) else RotRecord.from_dict(record) for record in records]
        # cycle(typed) with max_steps << len(typed) would otherwise train on only
        # the first max_steps records in id-sorted file order. Shuffle so the step
        # budget samples a representative slice across all records (and any appended
        # data stays reachable). Disable with REGRPO_SHUFFLE_RECORDS=0.
        if os.environ.get("REGRPO_SHUFFLE_RECORDS", "1") == "1":
            random.shuffle(typed)
        environment = _rl_environment()
        self.env = environment.OfflineReplayEnvironment(
            typed,
            lambda_exec=self.lambda_exec,
            eta=self.eta,
            lambda_val=self.lambda_val,
        )
        metrics = []
        # Skip records that crash the processor (e.g. Qwen2-VL 'image features and image
        # tokens do not match' — a minority of base records have corrupt image-token counts
        # that the first-250 window never hit). Count only successful steps so the budget holds.
        cycle = itertools.cycle(typed)
        good = 0
        skipped = 0
        while good < max_steps:
            record = next(cycle)
            try:
                result = self.train_step(record, int(record.rot_meta.step_index))
            except Exception as exc:  # noqa: BLE001 — skip the bad record, keep training
                skipped += 1
                if skipped % 25 == 1:
                    print(f"[skip] train_step failed ({skipped} skipped): {repr(exc)[:140]}")
                if skipped > max(50, max_steps * 5):
                    raise RuntimeError(f"too many bad records ({skipped}); aborting") from exc
                continue
            result["step"] = good
            print(result)
            metrics.append(result)
            good += 1
        print(f"[fit] completed {good} steps, skipped {skipped} bad records")
        self.save()
        return metrics

    def save(self) -> None:
        """Persist the trained policy adapter and processor."""

        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self.policy.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"saved ReGRPO adapter to {self.output_dir}")

    def _prepare_inputs(self, full_text: str, messages: list[dict]) -> Any:
        image_paths = _record_image_paths_from_messages(messages)
        if image_paths:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None
        return self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _token_logprobs(self, model: Any, inputs: Any) -> Any:
        import torch

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    if args.data:
        config["data"] = args.data
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    import torch

    random.seed(int(config.get("seed", 13)))
    torch.manual_seed(int(config.get("seed", 13)))
    records = read_json(config["data"])
    trainer = QwenVLReGRPOTrainer(
        model_name=config.get("model_name", "Qwen/Qwen2-VL-7B-Instruct"),
        lambda_exec=float(config.get("lambda_exec", 1.0)),
        eta=float(config.get("eta", 0.1)),
        lambda_val=float(config.get("lambda_val", 0.0)),
        two_stream_advantage=bool(config.get("two_stream_advantage", False)),
        verifier_alpha=float(config.get("verifier_alpha", 1.0)),
        beta=float(config.get("beta", 0.04)),
        lr=float(config.get("lr", 5e-6)),
        use_lora=bool(config.get("use_lora", True)),
        device=str(config.get("device", "cuda")),
        max_len=int(config.get("max_len", 4096)),
        output_dir=config.get("output_dir"),
        init_adapter=config.get("init_adapter"),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        length_normalize=bool(config.get("length_normalize", False)),
        advantage_normalize=bool(config.get("advantage_normalize", False)),
        advantage_clip=_optional_float(config.get("advantage_clip")),
        advantage_estimator=str(config.get("advantage_estimator", "group_mean")),
        grad_clip_norm=_optional_float(config.get("grad_clip_norm")),
        warmup_steps=int(config.get("warmup_steps", 0)),
        image_min_pixels=_optional_int(config.get("image_min_pixels", DEFAULT_IMAGE_MIN_PIXELS)),
        image_max_pixels=_optional_int(config.get("image_max_pixels", DEFAULT_IMAGE_MAX_PIXELS)),
        freeze_modules_to_save=bool(config.get("freeze_modules_to_save", False)),
    )
    trainer.fit(records, int(config.get("max_steps", 1000)))


def _freeze_modules_to_save(policy: Any) -> None:
    """Freeze any ``modules_to_save`` adapter copies (e.g. lm_head) for RL.

    When the init adapter carries full-rank ``modules_to_save`` modules (the
    image-SFT adapter saves ``lm_head``), RL otherwise has to re-fit a ~545M
    full-rank head over only a few hundred offline contrastive steps, which
    under-trains relative to its surface. Freezing keeps the SFT-trained head at
    its value so RL adapts only the LoRA layers, matching the attn-only recipe
    that works. No-op when the adapter has no ``modules_to_save``.
    """

    frozen = 0
    for name, param in policy.named_parameters():
        if "modules_to_save" in name and param.requires_grad:
            param.requires_grad_(False)
            frozen += int(param.numel())
    if frozen:
        print(f"ReGRPO: froze modules_to_save ({frozen:,} params) for RL")


def _enable_policy_gradient_checkpointing(policy: Any) -> None:
    configs = []
    for model in (policy, getattr(policy, "base_model", None)):
        config = getattr(model, "config", None)
        if config is not None and id(config) not in {id(item) for item in configs}:
            configs.append(config)
    for config in configs:
        setattr(config, "use_cache", False)
    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    policy.enable_input_require_grads()


def _context_messages(record: RotRecord, step_index: int) -> list[dict]:
    count = 0
    for index, turn in enumerate(record.conversations):
        if turn.get("role") != "assistant":
            continue
        if count == step_index:
            return [dict(item) for item in record.conversations[:index]]
        count += 1
    return [dict(item) for item in record.conversations[:-1]]


def _qwen_vl_messages(
    record: RotRecord,
    context_messages: list[dict],
    *,
    image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    image_max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
) -> list[dict]:
    messages = [_normalize_message(turn) for turn in context_messages]
    image_paths = _record_image_paths(record)
    if not image_paths:
        return messages
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            text = _message_text(message.get("content", ""))
            content = [
                _qwen_vl_image_message(
                    path,
                    min_pixels=image_min_pixels,
                    max_pixels=image_max_pixels,
                )
                for path in image_paths
            ]
            content.append({"type": "text", "text": text})
            messages[index] = {"role": "user", "content": content}
            break
    return messages


def _qwen_vl_image_message(
    path: str,
    *,
    min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    max_pixels: int | None = DEFAULT_IMAGE_MAX_PIXELS,
) -> dict:
    item = {"type": "image", "image": path}
    if min_pixels is not None:
        item["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        item["max_pixels"] = int(max_pixels)
    return item


def _normalize_message(turn: dict) -> dict:
    role = turn.get("role", turn.get("from"))
    content = turn.get("content", turn.get("value", ""))
    return {"role": str(role), "content": content}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _record_image_paths(record: RotRecord | dict) -> list[str]:
    obj = record.to_dict() if isinstance(record, RotRecord) else record
    paths = _image_paths_from_value(obj.get("image"))
    meta = obj.get("rot_meta") or {}
    paths.extend(_image_paths_from_value(meta.get("image")))
    paths.extend(_image_paths_from_value(meta.get("images")))
    seen = set()
    unique = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            unique.append(path)
    return _usable_image_paths(unique)


def _image_paths_from_value(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item]
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            paths.extend(_image_paths_from_value(item))
        return paths
    return []


def _record_image_paths_from_messages(messages: list[dict]) -> list[str]:
    paths = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and item.get("image"):
                paths.append(str(item["image"]))
    return _usable_image_paths(paths)


def _usable_image_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if _is_usable_image_source(path)]


def _is_usable_image_source(path: str) -> bool:
    if path.startswith(("http://", "https://", "data:")):
        return True
    if os.path.exists(path):
        return True
    _note_missing_image(path)
    return False


def _note_missing_image(path: str) -> None:
    global _MISSING_IMAGE_WARNING_EMITTED, _SKIPPED_MISSING_IMAGES
    _SKIPPED_MISSING_IMAGES += 1
    if _MISSING_IMAGE_WARNING_EMITTED:
        return
    _MISSING_IMAGE_WARNING_EMITTED = True
    warnings.warn(
        "skipping missing local image files in Qwen-VL RL training; "
        f"first missing path: {path}",
        RuntimeWarning,
        stacklevel=3,
    )


def _candidate_token_boundaries(
    processor: Any,
    prefix: str,
    reflection: str,
    action: str,
    full_text: str,
) -> tuple[int, int, int, int]:
    tokenizer = _processor_tokenizer(processor)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    reflection_end = len(tokenizer(prefix + reflection, add_special_tokens=False).input_ids)
    action_end = len(tokenizer(prefix + reflection + action, add_special_tokens=False).input_ids)
    full_text_len = len(tokenizer(full_text, add_special_tokens=False).input_ids)
    return prefix_len, reflection_end, action_end, full_text_len


def _processor_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _tail_span_slices(
    prefix_len: int,
    reflection_end: int,
    action_end: int,
    *,
    offset: int = 0,
) -> tuple[slice, slice]:
    reflection_slice = _span_to_logprob_slice(prefix_len - offset, reflection_end - offset)
    action_slice = _span_to_logprob_slice(reflection_end - offset, action_end - offset)
    return reflection_slice, action_slice


def _tail_scored_token_bounds(
    total_tokens: int,
    reflection_tokens: int,
    action_tokens: int,
    end_tokens: int,
) -> tuple[int, int, int]:
    action_end = int(total_tokens) - int(end_tokens)
    reflection_end = action_end - int(action_tokens)
    prefix_len = reflection_end - int(reflection_tokens)
    return prefix_len, reflection_end, action_end


def _span_to_logprob_slice(start_token: int, end_token: int) -> slice:
    if end_token <= start_token:
        return slice(0, 0)
    return slice(max(start_token - 1, 0), max(end_token - 1, 0))


def _move_inputs(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


def _rl_core() -> Any:
    from regrpo.rl import core

    return core


def _rl_environment() -> Any:
    from regrpo.rl import environment

    return environment


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate_advantage_estimator(value: str) -> str:
    estimator = str(value)
    if estimator not in {"group_mean", "rloo"}:
        raise ValueError(f"unknown advantage_estimator: {value}")
    return estimator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
