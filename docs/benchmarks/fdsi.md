# FDSI benchmark

A reproducible run of `adapt_decomp` on the FDSI dataset: 100 synthetic HD-EMG recordings
(5 subjects × 5 contractions × 4 SNR levels), each with the ground-truth spike trains of its
simulated motor units ([Dataset](fdsi/dataset.ipynb)). It has three stages:

1. **Calibrate** every recording with CBSS on its first 5 s.
2. **Search** the adaptation hyperparameters on a pool of three recordings.
3. **Apply** the no-adaptation baseline and each search's winner to all 100 recordings.

The [Results](fdsi/report.ipynb) page reads the tables this produces.

[`benchmark.yaml`][spec] declares the whole experiment. A single CLI runs it, one command per
stage. Each stage is a list of independent tasks, so it runs the same way on a laptop (`--all`)
or as PBS Pro array jobs.

## Quick start

Run everything from the repository root, with the `adapt_decomp` environment active:

```sh
python scripts/download_data.py get fdsi_benchmark-data   # inputs, about 10 GB

python -m benchmarks.fdsi calibrate --all --n-workers 8
python -m benchmarks.fdsi search --all                    # each search uses 12 cores
python -m benchmarks.fdsi apply --all --n-workers 8
python -m benchmarks.fdsi collect                         # tables for the report
python -m benchmarks.fdsi import-v10                      # v1.0 comparison (needs fdsi_benchmark-outputs)
```

To check the pipeline in about 20 minutes, use the small spec:
`--spec benchmarks/fdsi/benchmark_smoke.yaml` (1 subject, 4 recordings, 4 trials per search).

## Commands

| Command | Does |
|---|---|
| `tasks STAGE [--count \| --status \| --ncpus]` | Lists a stage's tasks (index, id), or prints their number, whether each is current, or the cores one task uses. |
| `calibrate` | One task per recording: `CBSS(...).decompose()` on the first 5 s, then `select_supervised(roa_th=0.9)`. A recording with no matched unit is recorded as `skipped`. |
| `search` | One task per search: `optimize_adapt_decomp` on the pool, trimmed to the samples after the calibration window. |
| `apply` | One task per (config, recording): `AdaptDecomp.from_calibration(...).process_from_calib_end()`, then SIL and per-unit RoA. The configs are the fixed baseline and each search's winner. |
| `collect` | Gathers every current output into `tables/*.csv`. It is read-only and works on a partial run. |
| `import-v10` | Computes the same per-unit metrics from the cached v1.0 results (`tables/v1_0_units.csv`). |
| `verify STAGE --tasks 0,17` | Re-runs tasks into a scratch root and checks they reproduce: spike trains bit for bit, search trials within 1e-6. |

`calibrate`, `search` and `apply` take one of:

- `--task-index N`: one task;
- `--array-index N --chunk-size K`: tasks `[N·K, (N+1)·K)`. The array index defaults to
  `$PBS_ARRAY_INDEX`, then `$PBS_ARRAYID`;
- `--all`: every task, optionally over `--n-workers` processes.

## Settings

[`benchmark.yaml`][spec] holds:

- the recordings and the calibration settings (v1.0's);
- the pool: sub-01, 30 dB, triangular-ramp 40/10/5 s;
- five searches: `sv_loss` mean/sum, Pareto `wh_loss`/`sv_loss` mean/sum with the min-sv
  member, and `roa` (the ground-truth upper bound). All search `wh_learning_rate`,
  `sv_learning_rate` and `centroid_momentum`, with 50 trials and `n_jobs=4`;
- the fixed baseline ([`default_fixed.yaml`][fixed]).

The search settings use the keys of [`sweep_optuna.yaml`][sweep]. Each search's resolved
settings are saved as `search.yaml`, re-runnable with `scripts/run.py run_optuna`. The searches
use no unit selection, because the calibrations already keep only the units matched to the
ground truth.

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
       recordings.csv, units.csv, provenance.csv, v1_0_units.csv, collect.meta.yaml
provenance/patches/<hash>.patch
```

Each table's rows:

| Table | One row per | Holds |
|---|---|---|
| `calibrations.csv` | recording | status, units, mean calibration RoA, units with SIL ≥ 0.9 |
| `calibration_units.csv` | calibrated unit | `recording`, `unit`, `gt_unit` (the simulated motor unit it tracks), `roa_calib`, `sil_calib`, `cov_isi_calib` |
| `searches.csv` | search trial | parameters, objective values, pooled losses and RoA |
| `best_configs.csv` | search | the chosen trial and its parameters |
| `recordings.csv` | (config, recording) | status, units, mean RoA, losses, time per batch |
| `units.csv`, `v1_0_units.csv` | (config, recording, unit) | `gt_unit`, `roa_calib`, RoA over the whole recording, after calibration and per triangular phase, SIL, spike counts |
| `provenance.csv` | task | status, run time, host, CPU, commit, dirty |

## Run metadata

Every output has a `*.meta.yaml` next to it. It records:

- **The task:** stage, index, id, cache key, status, and the spec used.
- **Inputs and outputs:** input files with their SHA-256 and upstream keys; the output files
  and an output digest (e.g. the spike trains' SHA-256).
- **Timing:** `started_at`, `finished_at` and `run_time_s`.
- **The machine:** host (hostname, PBS job id and array index); OS; hardware (CPU model and
  cores, cores available, torch threads, GPUs, memory total/limit/free); Python and package
  versions, torch's BLAS backend and the thread variables.
- **`git`:** remote, commit, branch, `dirty`, the patch, and untracked files.
- **`command`:** the command as invoked.
- **`reproduce`:** the lines that redo exactly this output, for example:

```text
git clone https://github.com/imendezguerra/adapt_decomp.git
cd adapt_decomp
git checkout -b "benchmark_v1_1-apply-17" <commit>
git apply <outputs_root>/provenance/patches/<hash>.patch     # only when run from a dirty tree
conda env create -f environment.yaml
conda activate adapt_decomp
pip install -e .
python scripts/download_data.py get fdsi_benchmark-data
python -m benchmarks.fdsi apply --spec benchmarks/fdsi/benchmark.yaml --task-index 17
```

A run from a tree with uncommitted changes still proceeds. Its `git diff HEAD` is saved once
per distinct diff under `provenance/patches/`, and `git apply` is added to the reproduce lines.
Untracked files are listed but not saved, so commit them before a run that matters.

## Reproducibility

Every run uses one torch/BLAS thread: `python -m benchmarks.fdsi` sets `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to 1. A search uses
`n_cores = n_jobs × pool size`, so each of its runs also gets one thread. Results therefore
don't depend on the machine's core count.

On the same environment (`environment.yaml`), CPU outputs reproduce bit for bit, which `verify`
checks. Across BLAS libraries (MKL vs OpenBLAS) single runs differ in the last digits, which can
occasionally reorder close search trials.

## PBS Pro

```sh
bash benchmarks/fdsi/pbs/submit.sh --dry-run    # print the qsub commands
bash benchmarks/fdsi/pbs/submit.sh              # calibrate -> search -> apply -> collect
bash benchmarks/fdsi/pbs/submit.sh --from apply # resubmit from a stage
```

[`submit.sh`][submit] sizes each array from `tasks STAGE --count`. Each stage waits for the
previous one with `-W depend=afterok`. The per-stage memory, walltime and chunk size are at the
top of the script. Add your cluster's module and conda lines to [`stage.pbs`][stage].

## Release checklist

The [Results](fdsi/report.ipynb) page is rendered from the report notebook's stored outputs,
which come from the full run:

1. Commit everything (every output records its commit and whether the tree was dirty).
2. On the cluster: `bash benchmarks/fdsi/pbs/submit.sh`; when it finishes, run
   `python -m benchmarks.fdsi import-v10` and `verify` a few tasks of each stage, e.g.
   `python -m benchmarks.fdsi verify apply --tasks 0,250,599`.
3. Execute the report against the full tables and keep its outputs:
   `jupyter nbconvert --to notebook --execute --inplace benchmarks/fdsi/report.ipynb`.
   Remove its "Results pending" note, then commit it.
4. Push to `main`: the docs workflow rebuilds and redeploys the site.

[spec]: https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/benchmark.yaml
[fixed]: https://github.com/imendezguerra/adapt_decomp/blob/main/configs/adapt_configs/default_fixed.yaml
[sweep]: https://github.com/imendezguerra/adapt_decomp/blob/main/configs/sweep_configs/sweep_optuna.yaml
[submit]: https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/submit.sh
[stage]: https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/stage.pbs
