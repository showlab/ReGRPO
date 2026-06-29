#!/usr/bin/env bash
# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
#
# Stage-1 RoT-SFT CPU smoke (Qwen2.5-0.5B + LoRA, >=1 step).
# Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 to force fully offline runs.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m regrpo.sft.train_sft \
  --config regrpo/configs/sft_smoke.yaml \
  --max-steps "${1:-2}"
