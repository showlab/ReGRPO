#!/usr/bin/env bash
# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
#
# Stage-2 ReGRPO RL CPU smoke (offline contrastive update, >=1 step).
# Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 to force fully offline runs.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m regrpo.rl.trainer_minimal \
  --config regrpo/configs/regrpo_smoke.yaml \
  --data samples/rot_train.sample.json \
  --max-steps "${1:-1}"
