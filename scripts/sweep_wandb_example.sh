#!/usr/bin/env bash
# Wandb sweep example:set -e
cd "$(dirname "$0")/.."

python scripts/run.py run_wandb \
  --data_config configs/data_configs/fdsi_pool_memory_example.yaml \
  --adapt_config configs/adapt_configs/default_muniverse_lrfixed.yaml \
  --sweep_config configs/sweep_configs/sweep_wandb.yaml \
  --wandb_project_name adapt_decomp
