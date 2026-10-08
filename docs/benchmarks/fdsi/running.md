# Running the benchmark

The [FDSI benchmark](../fdsi.md)'s stages are functions in
[`benchmarks/fdsi/pipeline.py`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pipeline.py),
one per task, each calling the `adapt_decomp` API directly. The notebooks' `RUN` cells, the
command line and the PBS jobs all call them. To benchmark your own data, copy
`benchmarks/fdsi/`, replace `fdsi.py`'s paths and loaders with your dataset's, and edit
`config.yaml`.

## Commands

From the repository root, with the `adapt_decomp` environment active,
`python -m benchmarks.fdsi STAGE`:

| Stage | Tasks |
|---|---|
| `calibrate` | One per recording: `CBSS(...).decompose()` on the first 5 s, then `select_supervised(roa_th=0.9)`. A recording with no matched unit gets an empty table and nothing to apply. |
| `search` | One per search: `optimize_adapt_decomp` on the pool, its recordings' calibrations with the EMG and ground truth after the calibration window (`PooledDatasetMemory`). |
| `apply` | One per (config, recording): `AdaptDecomp.from_calibration(...).process_data()` after the calibration window, CBSS's own output prepended over it, then SIL and per-unit RoA. The configs are the fixed baseline and each search's winner: from the search's outputs, or else the published one in `results/<version>/configs/`. |
| `collect` | Gathers the scores into `benchmarks/fdsi/results/<version>/`, scoring any saved result that has none yet (e.g. downloaded). It works on a partial run. |

Options:

- `--workers N`: run N calibrate or apply tasks at once (searches run one at a time, each on
  every core it is given);
- `--index I --chunk K`: run only tasks `[I·K, (I+1)·K)`, an array job's share. `--index`
  defaults to `$PBS_ARRAY_INDEX`;
- `--count`: print the stage's number of tasks;
- `--quick`: run `config.yaml`'s `quick` section, a small check of the whole pipeline, under
  `<version>-quick`;
- `--config PATH`: another config file.

A task whose output exists is skipped, so rerunning a stage only computes what's missing. To
recompute an output, delete it. Changing a setting means a new experiment: give it a new
`version`, so its outputs and results never mix with the old ones. Outputs are written to a
temporary name and renamed into place, so an interrupted task leaves nothing behind.

## Outputs and results

Under `data/fdsi_benchmark/outputs/<version>/`, git-ignored:

```text
calibration/<sub>/<stub>_cbss.pkl, _cbss_units.csv         the CBSSResult and its units
searches/<name>/trials.csv, best_config.yaml                every trial (the chosen one marked), the winner
results/<config>/<sub>/<stub>.npz, _units.csv               spikes, sources and matched gt_unit; the scores
```

The results, in git under `benchmarks/fdsi/results/<version>/`:

| File | One row per | Holds |
|---|---|---|
| `units.csv` | (config, recording, unit) | `gt_unit` (the simulated motor unit it tracks), `roa_calib`, RoA over the whole recording, after calibration and per triangular phase, SIL, spike counts |
| `calibration_units.csv` | calibrated unit | `gt_unit`, `roa_calib`, `sil_calib`, `cov_isi_calib` |
| `trials.csv` | search trial | the search, parameters, objective values, pooled losses and RoA, `chosen` |
| `configs/<search>.yaml` | search | the chosen trial's `AdaptConfig` |
| `run.yaml` | | the adapt_decomp version, commit, date, data DOI, config labels and the whole config |

Releasing a version publishes its `results/` and `searches/` outputs as
`fdsi_benchmark-outputs-<version>`; the calibrations, which embed the raw EMG, are recomputed
from the data.

## Reproducibility

Every calibration and application runs on one torch and BLAS thread, so its results don't
depend on the machine's core count. A search spreads its runs over the cores it is given
(`n_cores` in `config.yaml`, every core available by default), which sets only its speed: the
suggested settings depend on the seed and `n_jobs`, and on FDSI the spikes of a run are
identical on 1 to 32 threads. On the same environment (`environment.yaml`), CPU outputs
reproduce bit for bit. Across BLAS libraries (MKL vs OpenBLAS) single runs differ in the last
digits, which can occasionally reorder close search trials.

## PBS Pro

```sh
bash benchmarks/fdsi/pbs/submit.sh --dry-run    # print the qsub commands
bash benchmarks/fdsi/pbs/submit.sh              # calibrate -> search -> apply -> collect
bash benchmarks/fdsi/pbs/submit.sh --from apply # resubmit from a stage
bash benchmarks/fdsi/pbs/submit.sh --quick      # the quick check
```

[`submit.sh`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/submit.sh)
sizes each array job from `STAGE --count`, and each stage waits for the previous one with
`-W depend=afterok`. The per-stage cores, memory, walltime and chunk size are at the top of the
script. Add your cluster's module and conda lines to
[`stage.pbs`](https://github.com/imendezguerra/adapt_decomp/blob/main/benchmarks/fdsi/pbs/stage.pbs).
Job logs go to `.job_outputs/`.
