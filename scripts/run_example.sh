#!/usr/bin/env bash
# Plain run example
set -e
cd "$(dirname "$0")/.."

python scripts/run.py run \
  --data_config configs/data_configs/fdsi_example.yaml \
  --adapt_config src/adapt_decomp/adaptation/presets/muniverse.yaml \
  --wandb_project_name adapt_decomp
