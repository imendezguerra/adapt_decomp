# Hyperparameter optimisation

The learning rates and the centroid momentum set how fast the [adaptation](adaptation.md)
follows changes in the EMG. Too slow, and units are lost when the contraction changes; too
fast, and the model follows noise or drifts onto another unit. `optimize_adapt_decomp`
chooses them for a dataset with an [Optuna](https://optuna.org) search over a pool of
calibrated recordings. Each trial adapts every recording with one suggested setting and scores
it with the adaptation's losses, which need no ground truth.

```python
from adapt_decomp.adaptation import AdaptConfig
from adapt_decomp.adaptation.optimize import optimize_adapt_decomp

result = optimize_adapt_decomp(
    pool=pool,
    objectives="sv_loss",
    base_config=AdaptConfig.from_preset("muniverse"),
    n_trials=50,
)
result.best_config.to_yaml("tuned_config.yaml")
```

## How it works

```mermaid
flowchart LR
    P["Pool of calibrated\nrecordings"] --> U["Optional unit\nselection"]
    U --> T["Trial: suggest a setting,\nadapt every recording"]
    T --> S["Score: losses\nsummed over the pool"]
    S -->|"sampler (TPE)"| T
    S --> B["Best trial, or the\nPareto front"]
    B --> R["OptimisationResult\n(best_config, study)"]
```

## Key parameters

| Argument | Default | Controls | Change it |
|---|---|---|---|
| `pool` | (required) | The calibrated recordings every trial adapts | See [the pool](#the-pool) |
| `base_config` | `AdaptConfig()` | The config every trial starts from; the search overrides the searched fields | To the closest [preset](adaptation.md#presets) |
| `param_space` | `DEFAULT_PARAM_SPACE` | The fields searched and their ranges | To search more fields, or narrower ranges; see [what is searched](#what-is-searched) |
| `objectives` | `"sv_loss"` | What each trial is scored on; two or more give a Pareto front | See [objectives](#objectives) |
| `selection` | `"min_sv_loss"` | Which Pareto front member builds `best_config` | See [Pareto front](#two-objectives-a-pareto-front) |
| `unit_selection` | None | Which calibration units the search adapts and scores | `"unsupervised"` without ground truth; see [unit selection](#unit-selection) |
| `unit_selection_kwargs` | `{"cov_th": 0.3}` | The thresholds of `unit_selection="unsupervised"` | |
| `compute_roa` | False | Whether to also score every trial against the ground truth | True on simulations, to compare objectives |
| `n_trials` | 100 | Number of trials | Fewer for a quick check; 50 or more in practice |
| `n_jobs` | 1 | Trials suggested together, before the sampler sees their results | Larger to run more trials at once; part of the search's definition |
| `n_cores` | None (all available) | Cores used; sets only the speed | To share a machine; see [speed and resources](#speed-and-resources) |
| `random_seed` | 1909 | Seed of the default sampler | None for a different search on every run |
| `sampler` | Multivariate TPE | The Optuna sampler | To use another Optuna sampler |
| `best_result_path` | None | Folder the best results, configs and study are written to as the search runs | To keep the results, including the completed trials of an interrupted search |
| `on_trial` | None | Function called after every trial with its losses | To stream progress to a tracker or a log |

The [API reference](../reference/optimisation.md) describes every argument.

## The pool

The pool is a dict from a recording's name to its data: the EMG, its calibration and the
calibration's config, plus its ground truth when there is one. `load_pooled_cbss_memory`
builds it from a list of files:

```python
from adapt_decomp.utils import load_pooled_cbss_memory

pool = load_pooled_cbss_memory(
    {
        "root": ".",
        "datasets": [
            {
                "name": "recording-1",
                "path_emg": "data/.../recording-1_emg.npz",
                "path_calib": "outputs/recording-1_cbss.pkl",  # CBSSResult.save()
                "path_calib_config": "outputs/recording-1_cbss_config.yaml",  # CBSSConfig.to_yaml()
                "path_gt": "data/.../recording-1_spikes.npz",  # optional
                "start": 10240,  # optional: adapt and score from this sample, e.g. the calibration's end
            },
            # one entry per recording
        ],
    }
)
```

`start` and `stop` select the samples every trial adapts and scores, EMG and ground truth alike
(the ground truth is matched to the calibration first, over the calibration window, which must
start at the recording's first sample). When `start` is the calibration's end, set
`source_fifo_from_calib=True` on the base config, as for
[adapting from the calibration end](../how-to/adapt-from-calibration-end.md).

A single recording is a one-entry pool. A pool too large for memory can use
`load_pooled_cbss_disk` instead, with the same list: each recording is then loaded for each
trial and released after it.

## What is searched

By default, `DEFAULT_PARAM_SPACE`:

| Field | Range | Scale |
|---|---|---|
| `wh_learning_rate` | 1e-4 to 5e-2 | log |
| `sv_learning_rate` | 1e-4 to 1e-1 | log |
| `centroid_momentum` | 0.1 to 0.9 | linear, in steps of 0.1 |

Every other field comes from `base_config`. To search more fields, extend the space, e.g.
`param_space={**DEFAULT_PARAM_SPACE, "batch_ms": ("int", 50, 200)}`. Each entry is
`(kind, low, high)` with kind `"log_float"`, `"float"` or `"int"`, or `("categorical", choices)`.
`"float"` and `"int"` take an optional step, `(kind, low, high, step)`, to draw only `low`,
`low + step`, ..., `high`.

To start a search from known parameter sets (e.g. a previous search's winner), pass
`initial_params=[{"wh_learning_rate": 0.036, "sv_learning_rate": 0.0053, "centroid_momentum": 0.9}]`.
They run first, in order, as the first start-up trials, so the sampler draws that many fewer
random ones. Each set gives every parameter of the space, inside its range.

## Objectives

| Objective | Measures | Ground truth |
|---|---|---|
| `"sv_loss"` (default) | How far the units' sources drift from calibration | Not needed |
| `"wh_loss"` | How far the whitening drifts from calibration | Not needed |
| `"total_loss"` | Their sum, in which `wh_loss` dominates | Not needed |
| `"roa"` | 100 minus the mean rate of agreement with the ground truth: the best a search could do | Needed |

Each objective is summed over the pool. `sv_loss` is first averaged over each recording's units
(`AdaptConfig.sv_loss_reduction="mean"`), so every recording weighs the same whatever its
number of units. `"sv_loss"` is the recommended single objective. `compute_roa=True` also
scores every trial against the ground truth, for comparison, without searching on it.

## Two objectives: a Pareto front

With two or more objectives, e.g. `objectives=("wh_loss", "sv_loss")`, the search keeps every
trial that no other trial beats on all of them: the Pareto front. `selection` picks the member
that builds `best_config`:

| `selection` | Picks | Ground truth |
|---|---|---|
| `"min_sv_loss"` (default) | The member with the lowest `sv_loss` | Not needed |
| `"knee"` | The member where improving one objective starts to cost the most of the other (two objectives only) | Not needed |
| `"max_roa_mean"` | The member with the highest mean rate of agreement (needs `compute_roa=True`) | Needed |
| a function | Your own rule, given the front's trials | |

`result.pareto_front` holds the front. To pick another member without searching again, use
`SELECTION_RULES["knee"](result.pareto_front)`.

## Unit selection

`unit_selection` sets which calibration units the search adapts and scores:

- `None` (default): every unit;
- `"unsupervised"`: the regularly firing units (CoV-ISI ≤ 0.3, set with
  `unit_selection_kwargs`), whose losses are reliable. Recommended without ground truth;
- `"supervised"`: the units matched to the ground truth.

The search only chooses the setting: applying `best_config` afterwards adapts every unit.

## Speed and resources

- `n_jobs` (default 1) is the number of trials suggested together, before the sampler sees
  their results. It is part of the search's definition.
- `n_cores` (default: every core the process may use, including a SLURM or PBS allocation)
  only sets the speed. Runs are spread over worker processes first, one recording each, and
  leftover cores become threads within each run.

Before it starts, the search predicts its peak memory from the pool and raises if it exceeds
the machine's or the job's limit, naming the largest `n_cores` that fits.

## What you get back

An `OptimisationResult`:

| Field | Holds |
|---|---|
| `best_config` | The base config with the chosen trial's setting |
| `study` | The Optuna study, with every trial |
| `pareto_front` | The front's trials, for a Pareto search; None otherwise |
| `outputs` | The chosen trial's `AdaptationResult` per recording, with `best_result_path` and an in-memory pool; None otherwise |

With `best_result_path`, results are written as the search runs: the best trial's results and
config (or one folder per front member), and `study.pkl` after every trial, so an interrupted
search keeps its completed trials.

## Reproducibility

With `random_seed` fixed, the suggested settings depend only on the seed and `n_jobs`, not on
the machine. `n_cores` sets the threads per run, which can change the last digits of the
losses. Keep the search's arguments with its output: the saved config and study don't record
them.

## Recipes

- [Tune hyperparameters](../how-to/tune-hyperparameters.md)
- [Use the command line](../how-to/use-the-command-line.md)
- [Run on a cluster](../how-to/run-on-a-cluster.md)
- [Plot results](../how-to/plot-results.md#searches)
