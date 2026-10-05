# Changelog

All notable changes to `adapt_decomp` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Pre-commit hooks (whitespace/YAML/TOML checks, ruff lint + format, dependency-spec sync
  check, fast tests on push) and GitHub Actions CI: tests on Python 3.10–3.12 across
  Linux/macOS/Windows, a minimum-dependency-versions job, and the pinned environment on all
  three OSes.
- Cross-platform reproducibility test (`tests/reproducibility/`): the tutorial's adaptation is
  checked against a stored reference on every OS.
- `Makefile` with common tasks.
- `AdaptDecomp.process_from_calib_end(emg, calib_indices, backward=False)`: keeps the
  calibration's own output over its window and adapts forwards from its last sample, and
  optionally backwards from its first sample (offline). `emg` is filtered once for both passes;
  each pass starts with FIFOs seeded from the samples next to it
  (`AdaptConfig.backward_fifo_seed`: `"forward_head"` or `"calib_tail"`).
- `process_data(..., reverse=True)`: offline backward pass, adapting from the last batch to
  the first; spikes/sources are returned in sample order.
- `AdaptConfig.source_fifo_from_calib`: seed the source FIFO with the calibration's tail, for
  online EMG that starts where calibration ends. `Decomposition.seed_fifos()` seeds either
  FIFO from given rows.
- `AdaptConfig.sv_loss_reduction` (`"mean"` | `"sum"`): how `sv_loss_total` reduces across
  units per batch.
- `optimize_adapt_decomp`: one search entry point for in-memory or on-disk pools and one or
  several objectives, returning an `OptimisationResult`. New options:
  - `unit_selection` (`None` by default, every unit; `"unsupervised"`, CoV-ISI ≤ 0.3 via
    `unit_selection_kwargs`, recommended for recordings without ground truth; or
    `"supervised"`): which calibration units a search adapts and scores;
  - `selection` (`"min_sv_loss"`, `"knee"`, `"max_roa_mean"`, or a callable): which Pareto
    front member builds the best config;
  - `n_cores` (default: all physical cores available, SLURM- and PBS-aware): spread runs over worker
    processes first, leftover cores as torch threads per run (`plan_resources`); sets only
    the speed;
  - `n_jobs` now means trials suggested together (a synchronous batch), so the suggested
    parameters depend on `random_seed` and `n_jobs`, not on the machine; the default
    sampler's random start-up trials always run as one batch;
  - a memory check before any worker starts: predicted from each dataset's shape, raising
    over the machine's or job's limit and warning over the memory currently free, with
    guidance on what to change. Results are not bit-identical to one-thread runs.
- `CBSSResult.unsupervised_mask()`: the quality-threshold mask behind `select_unsupervised()`.
- `scripts/run.py run_optuna` reads `objectives` (Pareto search from the CLI), `selection`,
  `unit_selection`/`unit_selection_kwargs`, `sampler` and `n_cores` from `--optim_config`.
- `adapt_decomp.utils.system`: `available_cores()`/`available_memory()` (moved from
  `adaptation/optimize/resources.py`, still re-exported from `adaptation.optimize`), now also
  honouring PBS (`NCPUS`, `PBS_NUM_PPN`) and cgroup v1 memory limits; `describe_system()` (host,
  scheduler job, OS, CPU/GPU/memory, Python and package versions).
- `adapt_decomp.utils.provenance`: `build_metadata()`/`write_metadata()`/`read_metadata()` for a
  result's metadata file (dates, run time, machine, git remote/commit/dirty state with a saved
  diff patch, the command, and the lines that reproduce it), plus `git_state()`.
- `benchmarks/fdsi/`: the FDSI benchmark as a spec (`benchmark.yaml`) and a CLI
  (`python -m benchmarks.fdsi`) with one command per stage (`calibrate`, `search`, `apply`,
  `collect`, `import-v10`, `verify`), content-hashed caching, a metadata file next to every
  output, and PBS Pro array-job scripts. Its tables include `calibration_units.csv` (one row per
  calibrated unit: the simulated motor unit it matches, `gt_unit`, and its calibration RoA, SIL
  and CoV-ISI), and every per-unit table carries `gt_unit`, so runs with different calibrations
  can be paired unit by unit. Stage code versions are part of the cache keys.
- `benchmarks/fdsi/report.ipynb` (with `benchmarks/fdsi/report.py`): the benchmark's results,
  read from the collected tables only; `benchmarks/fdsi/dataset.ipynb`: a tour of the raw data.
- `adapt_decomp.adaptation.optimize.front_mask()`: the Pareto front of a finished search's
  trials table.
- `plot_search_landscape`, `plot_search_front` and `plot_search_parameters`
  (`adapt_decomp.utils.plots`): static plots of a search, drawn from its trials table
  (`study.trials_dataframe()`) rather than a pickled study; `plot_metric_heatmap` takes `cmap`
  and `center`.
- A documentation site (MkDocs Material, `mkdocs.yml`), published to GitHub Pages from `main`:
  getting started, the guides, tested how-to guides (`docs/snippets/`), the rendered notebooks,
  the FDSI benchmark and an API reference generated from the docstrings. Build it with the new
  `docs` extra (`make docs`, `make docs-build`).

### Changed

- `environment.yaml` (exact, cross-platform pins) replaces the Windows-only
  `environment.lock.yaml`.
- `pyproject.toml` dependencies now have lower bounds; `pytest` and `ipykernel` moved to the
  `dev` extra. Minimum Python is 3.10 (the code already required it) and minimum torch is 2.7
  (earlier versions lack `slogdet` on Apple's MPS backend).
- Codebase formatted and linted with ruff.
- `calibrate_and_process` adapts from the end of the calibration window by default
  (`adapt_from="calib_end"`), returning CBSS's output over the window;
  `adapt_from="emg_start"` keeps the 1.0.0 behaviour.
- `sv_loss_total` is the per-unit mean by default (`sv_loss_reduction="mean"`), so pooled
  searches no longer favour recordings with more units. Not comparable with 1.0.0 values;
  set `"sum"` to reproduce them.
- The default Optuna sampler is a multivariate `TPESampler` (with `constant_liar` when
  `n_jobs > 1`), and `DEFAULT_PARAM_SPACE` includes `centroid_momentum` (0–0.95).
- Search results are staged through `<best_result_path>_temp`, and `study.pkl` is
  snapshotted after every trial in single-objective searches too.
- `adaptation/optimize.py` is now the subpackage `adaptation/optimize/`, one file per purpose
  (`search`, `scoring`, `units`, `resources`, `workers`, `pareto`, `persistence`,
  `deprecated`). Public imports from `adapt_decomp.adaptation.optimize` are unchanged.
  Internal helpers moved, and those shared between files dropped their leading underscore
  (e.g. `_score_dataset` is now `optimize.scoring.score_dataset`). The Pareto-front rules are
  public as `SELECTION_RULES`. Log records name the submodule; a loguru filter on
  `adapt_decomp.adaptation.optimize` still matches them.
- `ci/check_deps_sync.py` checks every pyproject extra (`dev`, `docs`) against
  `environment.yaml`, including its `pip:` entries, instead of a fixed list of names.

### Deprecated

- `optimize_adapt_decomp_pooled_memory`, `optimize_adapt_decomp_pooled_disk` and their
  `_pareto` variants: thin wrappers over `optimize_adapt_decomp` (with `unit_selection=None`)
  keeping their 1.0.0 signatures and return shapes; they emit a `FutureWarning`.

### Fixed

- The separation-vector convergence check compared each update with itself, so
  `sv_epochs > 1` always stopped after the first epoch.
- `load_example`'s legacy MATLAB calibrations stored `cov_isi` in percent; it is now a fraction,
  like CBSS's, so `cov_th` filters apply to them correctly.
- `adapt_decomp.__all__` listed six functions as objects rather than names, so
  `from adapt_decomp import *` raised `TypeError`.

## [1.0.0] - 2026-09-22

Full architectural rewrite of the package around two self-contained
subsystems, plus an hyperparameter optimisation, a test suite,
and a streaming/online decomposition. This is a breaking release: import
paths, config formats, and the CLI have all changed (see Migration below).

### Added

- **`cbss/` subpackage** — one-off calibration (convolutive blind source
  separation), independent of `adaptation/`: `CBSS` pipeline, `CBSSConfig`,
  `CBSSResult` with unit subsetting/selection (`select_unsupervised`,
  `select_supervised`), fixed-point ICA, PCA/whitening helpers.
- **`adaptation/` subpackage** — per-batch online adaptation, rebuilt around
  a precalibrated `Decomposition` model + explicit running state:
  - `AdaptDecomp` with three construction paths: `__init__` (calibration
    from anything), `from_calibration(calibration, cbss_config, ...)` (from
    an existing `CBSSResult`), and `calibrate_and_process(emg, ...)` (runs
    `CBSS` and adaptation in one call, returning
    `(AdaptationResult, CBSSResult)`).
  - Two processing modes: `"offline"` (whole recording preprocessed
    upfront) and `"online"` (each raw batch filtered/centred/extended as it
    arrives), both driven batch-by-batch by `process_batch`, which can also
    be called directly for unbounded streaming use.
  - Covariance-estimation stability improvements: FIFO buffering, shrinkage,
    PCA prewhitening, Toeplitz-structured extension.
  - Included adaptive error-scaled learning rate for the whitening/separation
    updates (although fixed-scale learnining rates is still the recommended approach).
  - HDF5 save/load for decomposition outputs and per-batch parameters
    (`io.py`).
- **Hyperparameter optimisation (`adaptation/optimize.py`)** — Optuna-based
  search, pooled across one or more datasets:
  - `optimize_adapt_decomp_pooled_memory` / `optimize_adapt_decomp_pooled_disk`
    for single objective search (in-memory vs. on-disk `CBSSResult` pools).
  - `optimize_adapt_decomp_pooled_memory_pareto` /
    `optimize_adapt_decomp_pooled_disk_pareto` for Pareto multi-objective
    search, with a configurable selection rule over the resulting front (recommended approach with whitening and separation vector losses).
- **`scripts/run.py` CLI rewritten on `typer`**, with three subcommands:
  `run` (a single plain decomposition pass), `run_optuna` (Optuna search
  within one run), `run_wandb` (a wandb-managed sweep of plain runs, no
  nested Optuna). Example shell scripts added under `scripts/`.
- **Test suite** (`tests/`, mirroring the package layout) — previously
  untested; now includes unit tests for `adaptation/ops.py`,
  `data_structures.py`, `config.py`, `optimize.py` (including the Pareto
  path), `cbss/`, `preprocessing/`, `spikes/`, and `utils/loaders.py`, plus
  architecture/import boundary checks. Run with
  `pytest tests/test_backend.py -q`.
- **New configs**: `configs/sweep_configs/` (Optuna and wandb search
  settings), pooled/grid data configs for the FDSI benchmark
  (`configs/data_configs/fdsi_*`), and per dataset `adapt_configs/` defaults
  (fixed, forearm, muniverse, neuromotion, wrist).
- **New notebooks** under `notebooks/fdsi_benchmark/`: dataset overview,
  calibration, decomposition without adaptation, hyperparameter optimisation (Pareto,
  separation vector loss, rate of agreement), application, and a
  cross-method comparison notebook.
- **Documentation** under `docs/`: `architecture.md`, `calibration.md`,
  `adaptation.md`, `optimisation.md` — task-oriented guides linked from the
  README, replacing the old single package structure section.
- New dependencies: `pandas`, `scikit-learn`, `pyyaml`, `pytest`, `optuna`,
  `cmaes`, `loguru`, `plotly`, `tqdm`, `typer`.

### Changed

- **Package layout**: the flat top-level module set (`config.py`,
  `data_structures.py`, `io.py`, `loaders.py`, `plots.py`,
  `preprocessing.py`, `utils.py`) is replaced by `cbss/`, `adaptation/`,
  `preprocessing/`, `spikes/`, and `utils/` subpackages (see
  `docs/architecture.md`). Packaging switched from an explicit
  `packages = ["adapt_decomp"]` list to
  `[tool.setuptools.packages.find]` (auto-discovery).
- **Configs**: `configs/model_configs/` renamed to `configs/adapt_configs/`;
  data configs moved from `.yml` to `.yaml`.
- **Environment file**: `environment.yml` replaced by a pinned
  `environment.lock.yaml`.
- **CLI flag renamed**: `--model_config` is now `--adapt_config` (mirrors
  the `AdaptConfig`/`adapt_config` naming used throughout `adaptation/`).
- **Tutorial** moved from `tutorials/adaptive_emg_decomp_dyn_example.ipynb`
  to `notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb`.
- Version bumped `0.1.0` → `1.0.0`.

### Removed

- Top-level `config.py`, `data_structures.py`, `io.py`, `loaders.py`,
  `plots.py`, `preprocessing.py`, `utils.py` (superseded by the subpackages
  above).
- `configs/model_configs/sweep_loss.yml` (superseded by
  `configs/sweep_configs/`).

### Migration from 0.1.0

- See `notebooks/original_tutorial/adaptive_emg_decomp_dyn_example.ipynb` for
  a worked example of porting old code to the new pipeline.
- Replace any `from adapt_decomp.config import ...` /
  `from adapt_decomp.loaders import ...` / etc. with the new subpackage
  paths (e.g. `from adapt_decomp.adaptation.config import ...`,
  `from adapt_decomp.utils import load_data`).
- Rename `--model_config` to `--adapt_config` in any CLI invocation, and
  update data configs from `.yml` to `.yaml`.
- Rebuild the conda environment from `environment.lock.yaml` (not
  `environment.yml`).
- A calibrated model is now a `CBSSResult`, built via `CBSS(...).decompose()`
  or loaded from disk, then passed into `AdaptDecomp.from_calibration(...)`
  — see `docs/calibration.md` and `docs/adaptation.md`.

## [0.1.0] - 2025

Initial release: single-module adaptive EMG decomposition (whitening +
source separation adapted online, spike detection) driven by a `wandb`-based
CLI (`scripts/run.py`), with fixed-format `data_configs`/`model_configs`
YAML files and a single walkthrough notebook
(`tutorials/adaptive_emg_decomp_dyn_example.ipynb`), as described in
[Mendez Guerra et al., JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf).
