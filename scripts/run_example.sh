#!/usr/bin/env bash
# Plain run example
set -e
cd "$(dirname "$0")/.."

python scripts/run.py run \
  --data_config configs/data_configs/fdsi_pool_memory_example.yaml \
  --adapt_config configs/adapt_configs/default_muniverse_lrfixed.yaml \
  --wandb_project_name adapt_decomp
