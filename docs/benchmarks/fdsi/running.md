# Running the benchmark

The [FDSI benchmark](../fdsi.md) is one CLI, `python -m benchmarks.fdsi`, run from the
repository root with the `adapt_decomp` environment active. Each stage is a list of
independent tasks, so it runs the same way on a laptop (`--all`) or as array jobs on a cluster.
To benchmark your own data, copy `benchmarks/fdsi/`, replace `fdsi.py`'s paths and loaders with
your dataset's, and edit the spec.

## Commands

| Command | Does |
|---|---|
| `tasks STAGE [--count \| --status \| --ncpus]` | Lists a stage's tasks (index, id), or prints their number, whether each is current, or the cores one task uses. |
| `calibrate` | One task per recording: `CBSS(...).decompose()` on the first 5 s, then `select_supervised(roa_th=0.9)`. A recording with no matched unit is recorded as `skipped`. |
| `search` | One task per search: `optimize_adapt_decomp` on the pool, its recordings' saved calibrations with the EMG and ground truth after the calibration window (`PooledDatasetMemory`). |
| `apply` | One task per (config, recording): `AdaptDecomp.from_calibration(...).process_data()` on the samples after the calibration window, with CBSS's own output prepended over it, then SIL and per-unit RoA. The configs are the fixed baseline and each search's winner. |
| `collect` | Gathers every current output into `tables/*.csv`. It is read-only and works on a partial run. |
| `verify STAGE --tasks 0,17` | Re-runs tasks into a scratch root and checks they reproduce: spike trains bit for bit, search trials within 1e-6. |

`calibrate`, `search` and `apply` take one of:

- `--task-index N`: one task;
- `--array-index N --chunk-size K`: tasks `[N·K, (N+1)·K)`. The array index defaults to
  `$PBS_ARRAY_INDEX`, then `$PBS_ARRAYID`;
- `--all`: every task, optionally over `--n-workers` processes.

## Settings

[`benchmark.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/benchmark.yaml)
holds the recordings, the calibration settings, the pool, the searches and the fixed baseline
(the `fixed` preset). The search settings use the keys of
[`sweep_optuna.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_optuna.yaml),
and each search's resolved settings are saved as `search.yaml`, re-runnable with
`scripts/run.py run_optuna`. The searches use no unit selection, because the calibrations
already keep only the units matched to the ground truth.

## Caching

**Cache keys.** Every task has a cache key: a SHA-256 of everything its output depends on:

- the resolved settings, including the content of the config files they name;
- the SHA-256 of its input files;
- its upstream tasks' keys;
- the version of its stage's code (`STAGE_VERSIONS` in `spec.py`, bumped when a stage's
  outputs change).

The key never includes paths or the git commit. A task whose output carries the expected key is
skipped, so resubmitting a stage only computes what's missing.

**Stale outputs.** After a spec edit, the affected tasks are `stale`. They raise until re-run
with `--force`, so old results are never silently mixed with new ones. `tasks STAGE --status`
shows each task as `missing`, `done`, `skipped` or `stale`.

**Atomic writes.** Outputs are written to a temporary name and renamed into place. The metadata
file is written last and marks the task complete.

## Outputs

Under `outputs_root` (`data/fdsi_benchmark/outputs/benchmark_v1_1/`):

```text
calibration/<sub>/<stub>_cbss.pkl, _cbss_config.yaml, _cbss_units.csv, _cbss.meta.yaml
searches/<name>/study.pkl, trials.csv, best_config.yaml, base_config.yaml, search.yaml, best/
searches/<name>.meta.yaml
results/<config>/<sub>/<stub>.pkl, _config.yaml, _metrics.csv, .meta.yaml
tables/calibrations.csv, calibration_units.csv, searches.csv, best_configs.csv,
       recordings.csv, units.csv, provenance.csv, collect.meta.yaml
provenance/patches/<hash>.patch
```

| Table | One row per | Holds |
|---|---|---|
| `calibrations.csv` | recording | status, units, mean calibration RoA, units with SIL ≥ 0.9 |
| `calibration_units.csv` | calibrated unit | `recording`, `unit`, `gt_unit` (the simulated motor unit it tracks), `roa_calib`, `sil_calib`, `cov_isi_calib` |
| `searches.csv` | search trial | parameters, objective values, pooled losses and RoA |
| `best_configs.csv` | search | the chosen trial and its parameters |
| `recordings.csv` | (config, recording) | status, units, mean RoA, losses, time per batch |
| `units.csv` | (config, recording, unit) | `gt_unit`, `roa_calib`, RoA over the whole recording, after calibration and per triangular phase, SIL, spike counts |
| `provenance.csv` | task | status, run time, host, CPU, commit, dirty |

## Run metadata

Every output has a `*.meta.yaml` next to it, written with
[`build_metadata`](../../how-to/record-provenance.md). It records the task (stage, index,
cache key, spec), its inputs and outputs with their SHA-256, the timing, the machine, the git
state, the command, and the lines that reproduce exactly this output, for example:

```text
git clone https://github.com/imendezguerra/adapt_decomp.git
cd adapt_decomp
git checkout -b "benchmark_v1_1-apply-17" <commit>
git apply <outputs_root>/provenance/patches/<hash>.patch     # only when run from a dirty tree
conda env create -f environment.yaml
conda activate adapt_decomp
adapt-decomp-data get fdsi_benchmark-data
python -m benchmarks.fdsi apply --spec benchmarks/fdsi/benchmark.yaml --task-index 17
```

A run from a tree with uncommitted changes still proceeds. Its `git diff HEAD` is saved once
per distinct diff under `provenance/patches/`, and `git apply` is added to the reproduce lines.
Untracked files are listed but not saved, so commit them before a run that matters.

## Reproducibility

Every calibration and application uses one torch/BLAS thread: `python -m benchmarks.fdsi` sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1. A search uses
`n_cores = n_jobs × pool size × threads_per_run`: each run of a guided trial gets
`threads_per_run` torch threads, and the random start-up trials run that many more at once on a
thread each. Neither changes the suggested settings, and on FDSI the spikes of a run are
identical on 1, 2, 4 or 8 threads, so results don't depend on the machine's core count.

On the same environment (`environment.yaml`), CPU outputs reproduce bit for bit, which `verify`
checks. Across BLAS libraries (MKL vs OpenBLAS) single runs differ in the last digits, which can
occasionally reorder close search trials.

## PBS Pro

```sh
bash benchmarks/fdsi/pbs/submit.sh --dry-run    # print the qsub commands
bash benchmarks/fdsi/pbs/submit.sh              # calibrate -> search -> apply -> collect
bash benchmarks/fdsi/pbs/submit.sh --from apply # resubmit from a stage
```

[`submit.sh`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/submit.sh)
sizes each array from `tasks STAGE --count`. Each stage waits for the previous one with
`-W depend=afterok`. The per-stage memory, walltime and chunk size are at the top of the
script. Add your cluster's module and conda lines to
[`stage.pbs`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/stage.pbs).
