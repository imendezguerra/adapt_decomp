# Use the command line

`adapt-decomp`, installed with the package, calibrates, adapts and tunes from the shell. Each
command wraps the Python function it is named after, and its options are that function's
arguments, so the [API concepts](../guide/overview.md) describe both. The examples below run
on the example recording (`adapt-decomp data get fdsi_example-data`), from the directory that
holds `data/`.

| Command | Wraps | Does |
|---|---|---|
| `decompose` | `CBSS.decompose`, then `select_supervised` with `--gt` | Calibrates on a window and saves the calibration |
| `process_data` | `AdaptDecomp.from_calibration(...).process_data` | Adapts a saved calibration over a recording |
| `calibrate_and_process` | `AdaptDecomp.calibrate_and_process` | Both, in one call |
| `optimize_adapt_decomp` | `optimize_adapt_decomp` | Tunes the hyperparameters on a pool of recordings |
| `wandb_sweep` | `wandb.sweep` and `wandb.agent` | Tunes them with a wandb sweep instead (needs the `wandb` extra) |
| `data` | `download_data` | Lists and downloads the datasets (`adapt-decomp-data` is an alias) |

`adapt-decomp --help` lists the commands, and `adapt-decomp <command> --help` lists a
command's arguments and options, with their types and defaults.

## Input conventions

The commands are built with [Typer](https://typer.tiangolo.com), from the type hints of their
functions, and follow its conventions:

- **Arguments** are positional and come first: the EMG file of `decompose`, `process_data` and
  `calibrate_and_process`. **Options** are named, `--name value`, and come in any order.
  `--help` marks the required ones.
- **Option names** are the Python argument names, with their underscores: `--n_trials`,
  `--best_result_path`, `--source_fifo_from_calib`.
- **Flags** come in pairs: `--compute_roa` turns an option on, `--no-compute_roa` turns it off.
  Left out, the default applies; for `--source_fifo_from_calib`, the adaptation config's.
- **Lists**: repeat the option, as in `--objectives wh_loss --objectives sv_loss`.
- **Choices**: `--preset`, `--processing_mode`, `--adapt_from`, `--objectives`, `--emg_loader`
  and `--gt_loader` accept a fixed set of values, which `--help` lists.
- **Samples, not seconds**: `--start`, `--stop`, `--calib_start` and `--calib_stop` are sample
  indices, like `start` and `stop` in a [pool](../guide/optimisation.md#the-pool); 5 s at
  2048 Hz is 10240. A window includes its start and excludes its stop, like a Python slice.
- **The adaptation config** is `--preset NAME` or `--adapt_config FILE`, not both. With
  neither, `AdaptConfig()`'s defaults apply.
- **Paths** are relative to the current directory, except those inside a pool YAML, which are
  relative to its `root`.
- **Exit status**: 0 on success. On an error, the command prints the message and exits with a
  non-zero status, so commands chain in scripts and job files (`set -e`).

The files the commands read:

| Option | Format |
|---|---|
| The EMG argument | `.npz` with key `emg`, of shape (samples, channels); `--emg_loader neuromotion` for a NeuroMotion HDF5 file |
| `--gt` | `.npz` with key `spikes`: binary spike trains of shape (samples, units), or one array of spike indices per unit; `--gt_loader neuromotion` for a NeuroMotion HDF5 file |
| `--cbss_config`, `--adapt_config` | YAML, as written by `CBSSConfig.to_yaml()` or `AdaptConfig.to_yaml()`; fields left out keep their defaults |
| `--calibration`, `--calibration_config` | The `calibration.pkl` and `calibration_config.yaml` that `decompose` writes (`CBSSResult.save()`, `CBSSConfig.to_yaml()`) |
| `--data_config` | A [pool](../guide/optimisation.md#the-pool) YAML, e.g. [`configs/data_configs/fdsi_example.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/data_configs/fdsi_example.yaml) |
| `--search_config` | YAML of `optimize_adapt_decomp`'s search arguments, e.g. [`configs/sweep_configs/sweep_optuna.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_optuna.yaml) |

The outputs are the Python objects, saved: load them with `CBSSResult.load`,
`AdaptationResult.load` and `AdaptConfig.from_yaml`.

## Calibrate: `decompose`

```sh
EMG=data/fdsi_example/data/sub-01/noisy/sub-01_FDSI_triangular-ramp40s_snr30dB_emg.npz
GT=data/fdsi_example/data/sub-01/clean/sub-01_FDSI_triangular-ramp40s_spikes.npz

adapt-decomp decompose $EMG --stop 10240 --gt $GT --out_dir data/fdsi_example/outputs/cli
```

CBSS decomposes the first 5 s (`--start` 0 to `--stop` 10240) with `CBSSConfig()`'s defaults,
or the settings of `--cbss_config`. `--gt` keeps the units that match a ground-truth unit with
a rate of agreement of at least `--roa_th` (0.9). Without ground truth, set `selection:
unsupervised` and its thresholds (`selection_kwargs`) in the `--cbss_config` YAML instead (see
[Keeping the reliable units](../guide/calibration.md#keeping-the-reliable-units)). It writes
`calibration.pkl` and `calibration_config.yaml` into `--out_dir`.

## Adapt: `process_data`

```sh
adapt-decomp process_data $EMG \
  --calibration data/fdsi_example/outputs/cli/calibration.pkl \
  --calibration_config data/fdsi_example/outputs/cli/calibration_config.yaml \
  --preset muniverse --start 10240 --source_fifo_from_calib \
  --out data/fdsi_example/outputs/cli/adapted.pkl
```

It adapts the calibration over the samples from `--start` (here the calibration's end, with the
source FIFO seeded from it, as in
[Adapt from the calibration end](adapt-from-calibration-end.md)) and saves the
`AdaptationResult`. `--processing_mode online` preprocesses each raw batch as it would in real
time (see [Process online](process-online.md)).

## Both at once: `calibrate_and_process`

```sh
adapt-decomp calibrate_and_process $EMG --calib_stop 10240 --gt $GT --preset muniverse \
  --out_dir data/fdsi_example/outputs/cli-one-call
```

It calibrates on `--calib_start` to `--calib_stop`, keeps the matched units and adapts from the
end of the window. `adapted.pkl` covers the whole recording: CBSS's output over the window,
then the adapted samples. `--adapt_from emg_start` adapts from the first sample instead (see
[Where to start adapting](../guide/adaptation.md#where-to-start-adapting)). It also writes the
calibration and its config, for `process_data` with other settings.

## Tune: `optimize_adapt_decomp`

```sh
adapt-decomp optimize_adapt_decomp \
  --data_config configs/data_configs/fdsi_example.yaml \
  --search_config configs/sweep_configs/sweep_optuna.yaml \
  --preset muniverse --source_fifo_from_calib --n_trials 20 \
  --best_result_path data/fdsi_example/outputs/cli-search
```

Run it from the repository root: the example pool's paths are relative to it, and point to the
calibration the [quickstart](../getting-started/quickstart.md) saves. Its recordings are
adapted and scored from `start`, the calibration's end, hence `--source_fifo_from_calib`.

- `--search_config` holds the search's settings: the search space, `objectives`, `selection`,
  `unit_selection`, `n_trials`, `n_jobs`, `n_cores`, `random_seed`, `initial_params` and the
  TPE `sampler`'s arguments. Left out, `optimize_adapt_decomp`'s defaults apply.
- `--objectives`, `--n_trials` and `--n_cores` override the file for one run.
  `--compute_roa` scores every trial against the ground truth; by default it is on when every
  recording of the pool has ground truth.
- `--preset` or `--adapt_config` is the base config: every field the search does not tune.

It writes the search's results into `--best_result_path` as it runs (see
[What you get back](../guide/optimisation.md#what-you-get-back)), then `best_config.yaml`, the
chosen setting. Apply it with `process_data --adapt_config .../best_config.yaml`. For
arguments the command doesn't expose (`roa_kwargs`, another Optuna sampler, `on_trial`), call
`optimize_adapt_decomp` from Python.

## Log to wandb

With the `wandb` extra (`pip install "adapt-decomp[wandb]"`), `--wandb_project NAME` logs
`optimize_adapt_decomp` to [wandb](https://wandb.ai): one run for the whole search, with one
point per trial, and the chosen trial's per-batch losses for an in-memory pool. Set
`WANDB_MODE=offline` to log locally only, without an account.

`wandb_sweep` searches with a [wandb sweep](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)
instead of Optuna:

```sh
adapt-decomp wandb_sweep \
  --data_config configs/data_configs/fdsi_example.yaml \
  --sweep_config configs/sweep_configs/sweep_wandb.yaml \
  --preset muniverse --source_fifo_from_calib --wandb_project adapt_decomp
```

[`sweep_wandb.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_wandb.yaml)
is a wandb sweep config, plus `sweep_counts`, the number of runs (overridden by
`--sweep_counts`). Each run adapts the pool with the parameters wandb chose, set on the base
config.

| | `optimize_adapt_decomp` | `wandb_sweep` |
|---|---|---|
| Search | Optuna (multivariate TPE by default) | wandb (random, grid or Bayesian) |
| Objectives | One, or a Pareto front of several | One wandb metric |
| Unit selection, initial parameters, parallel trials | Yes | No |
| wandb | Optional (`--wandb_project`) | Needed, with an account unless offline |
| wandb runs | 1 for the whole search | 1 per trial |
| Per-batch losses logged | For the chosen trial | For every trial |
