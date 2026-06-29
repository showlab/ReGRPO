#!/usr/bin/env bash
# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
#
# Generate a small RoT sample from clean MAT trajectories with the teacher VLM.
# Requires OPENAI_API_KEY (teacher model defaults to gpt-4o; see configs/data_rot.yaml).
# Demos on the bundled clean sample; point --input at your full mat_train.json for scale.
#   usage: run_build_rot.sh [N_TRAJECTORIES] [OUTPUT_JSON]
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m regrpo.data.build_rot \
  --config regrpo/configs/data_rot.yaml \
  --input samples/mat_train.sample.json \
  --limit "${1:-50}" \
  --output "${2:-dataset/rot_train.generated.json}"
