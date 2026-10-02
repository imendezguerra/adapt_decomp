# FDSI benchmark, re-run with adapt_decomp v1.1

The FDSI benchmark of [`../fdsi_benchmark/`](../fdsi_benchmark) (v1.0), re-run on the same data
and calibrations with the v1.1 approach, and compared with the cached v1.0 results.

**What v1.1 changes** (all `lr_fixed`):
- **One search entry point:** `optimize_adapt_decomp`.
- **Sampler:** multivariate TPE.
- **Search space:** `centroid_momentum` is searched alongside the learning rates.
- **Loss:** per-unit-mean `sv_loss`.
- **Units:** only CoV-ISI ≤ 0.3 units are adapted and scored in the search.
- **Pareto selection:** knee selection, next to minimum `sv_loss`.
- **Where adaptation starts:** at the end of the 5 s calibration window, for both search and application (`process_from_calib_end`, CBSS's output kept over the window).

Every search runs with `n_jobs=1, n_workers=3` (`fdsi_v11.SEARCH`): trials are sequential, so the
results are reproducible, and each trial's three pooled recordings run in three processes.

## Running

1. Download the data and the v1.0 outputs (about 21 GB, into `data/fdsi_benchmark/`):

   ```sh
   python scripts/download_data.py get fdsi_benchmark
   ```

2. Run the notebooks in order, from this folder:

   | Notebook | Does | Writes |
   |---|---|---|
   | `02_fixed_adaptation` | No-adaptation baseline from the calibration end, all 100 recordings | `outputs/adaptation_v1_1/<sub>/fixed/` |
   | `03a_pareto_optimisation`, `03b_sv_loss_optimisation`, `03c_roa_optimisation` | The three pooled searches (50 trials each) | `outputs/adaptation_v1_1/optimisation/lr_fixed/<search>/`, `configs/adapt_configs/optim_muniverse_fdsi_v11_*.yaml` |
   | `04a_pareto_application` (min-sv and knee), `04b_sv_loss_application`, `04c_roa_application` | Apply each promoted config to all 100 recordings | `outputs/adaptation_v1_1/<sub>/lr_fixed/<sampler>/` |
   | `05_comparison_v1_0_vs_v1_1` | v1.0 vs v1.1 analysis (disk read only) | -- |

   Each notebook computes only what isn't cached yet (`RUN_* = True`); a rerun only loads.

## Reused, not recomputed

- **Data and calibrations:** v1.0's `00_dataset` and `01_calibration` aren't recreated. Both versions use the same cached calibrations, so they track exactly the same units and their results can be paired unit by unit.
- **v1.0 results:** read from `outputs/adaptation/` and never modified. v1.1 results use the same layout under `outputs/adaptation_v1_1/`, so every `../fdsi_benchmark/fdsi_common.py` path builder and aggregator works on either root. `fdsi_v11.py` only adds the v1.1-specific glue.
