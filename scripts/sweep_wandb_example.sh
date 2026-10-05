#!/usr/bin/env bash
# Wandb sweep example
set -e
cd "$(dirname "$0")/.."

python scripts/run.py run_wandb \
  --data_config configs/data_configs/fdsi_example.yaml \
  --adapt_config src/adapt_decomp/adaptation/presets/muniverse.yaml \
  --sweep_config configs/sweep_configs/sweep_wandb.yaml \
  --wandb_project_name adapt_decomp
