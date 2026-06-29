# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
"""Minimal CPU-friendly Stage-1 RoT-SFT trainer."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from regrpo.sft.dataset import RotSFTDataset, collate, load_sft_records


@dataclass
class SFTConfig:
    model_name: str
    clean_path: str
    rot_path: str
    clean_ratio: float
    max_len: int
    lr: float
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_steps: int
    batch_size: int
    seed: int
    device: str
    output_dir: str
    max_clean: int | None = None
    max_rot: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("SFT config must be a mapping")
        return cls(**data)


def train(config: SFTConfig) -> list[float]:
    """Run a tiny masked SFT loop and save the LoRA adapter/model."""

    _set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_sft_records(
        config.clean_path,
        config.rot_path,
        clean_ratio=config.clean_ratio,
        seed=config.seed,
        limit=None,
    )
    dataset = RotSFTDataset(
        records,
        tokenizer,
        max_len=config.max_len,
        clean_ratio=config.clean_ratio,
        seed=config.seed,
        max_clean=config.max_clean,
        max_rot=config.max_rot,
    )
    if len(dataset) == 0:
        raise ValueError("SFT dataset is empty")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
    device = torch.device(config.device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []
    iterator = iter(loader)
    for step in range(1, int(config.max_steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss_value}")
        loss.backward()
        optimizer.step()
        losses.append(loss_value)
        print(f"step={step} loss={loss_value:.6f}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved={output_dir}")
    return losses


def main() -> None:
    args = _parse_args()
    config = SFTConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    train(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
