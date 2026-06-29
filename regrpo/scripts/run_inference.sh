#!/usr/bin/env bash
# ReGRPO: Reflection-Augmented Group Relative Policy Optimization
# Copyright (c) 2026 Binjie Zhang @ Show Lab
# Licensed under the MIT License.
# This code references MAT-Agent (https://mat-agent.github.io/).
set -euo pipefail

cd "$(dirname "$0")/../.."
python -m regrpo.inference.agent --max-steps "${MAX_STEPS:-4}"
