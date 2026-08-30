#!/usr/bin/env bash
# Optuna search example
cd "$(dirname "$0")/.."

python scripts/run.py run_optuna \
  --data_config configs/data_configs/fdsi_pool_memory_example.yaml \
  --adapt_config configs/adapt_configs/default_muniverse_lrfixed.yaml \
  --optim_config configs/sweep_configs/sweep_optuna.yaml \
  --wandb_project_name adapt_decomp
