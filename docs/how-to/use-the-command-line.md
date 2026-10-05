# Use the command line

[`scripts/run.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/scripts/run.py) runs
an adaptation or a hyperparameter search from config files, logging to
[wandb](https://wandb.ai). It lives in the repository, so it needs a clone. The examples below
run on the example recording, with the calibration the [quickstart](../getting-started/quickstart.md)
saves; run them from the repository root.

Every command takes:

- `--adapt_config`: an `AdaptConfig` YAML, e.g. a preset in `src/adapt_decomp/adaptation/presets/`;
- `--data_config`: the recordings and their calibrations, e.g.
  [`configs/data_configs/fdsi_example.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/data_configs/fdsi_example.yaml),
  in the format of [the pool](../guide/optimisation.md#the-pool);
- `--wandb_project_name` (optional). Set `WANDB_MODE=offline` to log locally only.

## Adapt

```sh
python scripts/run.py run \
  --adapt_config src/adapt_decomp/adaptation/presets/muniverse.yaml \
  --data_config configs/data_configs/fdsi_example.yaml
```

Adapts every recording in the data config and logs its losses, timings and, with ground
truth, its rate of agreement.

## Search with Optuna

```sh
python scripts/run.py run_optuna \
  --adapt_config src/adapt_decomp/adaptation/presets/muniverse.yaml \
  --data_config configs/data_configs/fdsi_example.yaml \
  --optim_config configs/sweep_configs/sweep_optuna.yaml \
  --best_result_path data/fdsi_example/outputs/cli-search
```

Runs [`optimize_adapt_decomp`](../guide/optimisation.md) with the search settings in
[`sweep_optuna.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_optuna.yaml):
the search space, `objectives`, `selection`, `unit_selection`, `n_trials`, `n_jobs`, `n_cores`
and `random_seed`. `--objective` and `--n_trials` override the file for one run, and
`--best_result_path` saves the chosen trial's results. The whole study is one wandb run, with
one point per trial.

## Search with a wandb sweep

```sh
python scripts/run.py run_wandb \
  --adapt_config src/adapt_decomp/adaptation/presets/muniverse.yaml \
  --data_config configs/data_configs/fdsi_example.yaml \
  --sweep_config configs/sweep_configs/sweep_wandb.yaml
```

[`sweep_wandb.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_wandb.yaml)
is a [wandb sweep config](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration),
plus `sweep_counts`, the number of runs (overridden by `--sweep_counts`). Each run of the sweep
is a plain `run` with the parameters wandb chose.

## Which search

| | `run_optuna` | `run_wandb` |
|---|---|---|
| Search | Optuna (multivariate TPE by default) | wandb (random, grid or Bayesian) |
| Objectives | One, or a Pareto front of several | One wandb metric |
| wandb account | Not needed (`WANDB_MODE=offline`) | Needed |
| wandb runs | 1 for the whole study | 1 per trial |
| Per-batch losses logged | For the chosen trial, with `--best_result_path` | For every trial |
