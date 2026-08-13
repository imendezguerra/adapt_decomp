# Adaptive EMG decomposition in dynamic conditions based on online learning metrics with tunable hyperparameters

## Overview
This repository contains functions to adaptively decompose electromyography (EMG) into motor unit firings during dynamic conditions in real-time (~22 ms per 100 ms batch, CPU only with loss calculation) based on online learning metrics with tunable hyperparameters as described in [Mendez Guerra et al, JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf). The code is implemented in Python using PyTorch.

## Table of Contents
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Algorithm](#algorithm)
- [Package structure](#package-structure)
- [Command line interface](#command-line-interface)
- [Config reference](#config-reference)
- [Data loaders](#data-loaders)
- [Optimization](#optimization)
- [Changes from v1](#changes-from-v1)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Installation
To set up the project locally do the following:

1. Clone the repository:
    ```sh
    git clone https://github.com/imendezguerra/adapt_decomp.git
    ```
2. Navigate to the project directory:
    ```sh
    cd adapt_decomp
    ```
3. Create the conda environment from the `environment.yml` file:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the environment:
    ```sh
    conda activate adapt_decomp
    ```
5. Install the package:
    ```sh
    pip install -e .
    ```

Please note that the `environment.yml` only installs the `cpu` version of `pytorch`. To enable GPU acceleration, `cuda` will need to be installed manually (check command [here](https://pytorch.org/get-started/locally/)).

The code has been tested on macOS, Windows, and Linux.

## Tutorial
To learn how to use the adaptive decomposition go to [adaptive_emg_decomp_dyn_example](https://github.com/imendezguerra/adapt_decomp/blob/main/notebooks/examples/adaptive_emg_decomp_dyn_example.ipynb) for a step by step tutorial. The model requires a precalibrated decomposition model including extension factor, whitening matrix, separation vectors, spike and baseline centroids, EMG during calibration, and the resulting IPTs and spikes. To execute the code, download this [example contraction](https://imperiallondon-my.sharepoint.com/:f:/g/personal/im4417_ic_ac_uk/EkJvoEffPmdEnkoHeRItVt8BWyQd6kztbrszu6njnfHM0Q?e=wbbuZF) and save it in the repository directory under `data/JNE_data/sim`.

## Algorithm

**Whitening update (wh).**
At each batch, the extended EMG is pushed into a FIFO buffer and the whitened covariance `Rz` is re-estimated from the buffer using the current `wh`. The KL divergence between `Rz` and a reference (identity under `kl_to_identity`, or the calibration covariance under `kl_to_cal`) gives a scalar error `e_wh`, z-scored by its calibration standard deviation `sigma_K_cal`. The natural-gradient direction `(Rz − I) @ wh` (or its `kl_to_cal` analogue) is then scaled by `e_wh` and clipped so that `‖Δwh‖_F / ‖wh‖_F ≤ max_rel_delta_v`. This trust-region clip replaces a fixed learning rate and makes the step size scale-free across different contractions and electrode configurations. When `wh_b_coupling` is enabled, a first-order frame correction is also propagated to `sv` after each `wh` step to keep the separation matrix aligned with the new whitening frame.

**Source update (sv).**
Source signals `sources = Z @ sv.T` are computed after whitening. Spikes are detected via vectorised max-pool NMS on the source FIFO (providing left-edge context) and classified against adaptive centroids. A Tukey IQR gate excludes outlier spikes from adaptation. The log-cosh contrast `kappa` is computed over the current batch (`batch_based`) or only at spike times (`spike_based`). The contrast error `e_sv = kappa − kappa_cal` is z-scored and used to form a gradient that updates each row of `sv`. The update is clipped so that `‖Δsv‖_F / ‖sv‖_F ≤ max_rel_delta_b`, and `sv` is row-orthonormalised (QR decomposition). For the mathematical derivation see `VB_coupling_derivation.md`.

## Package structure
The package is composed of the following modules:
- `config.py`: Dataclass with all parameters for the decomposition adaptation.
- `data_structures.py`: `Data` (extended/preprocessed EMG dataset) and `Decomposition` (precalibrated model with adaptive state).
- `adaptation.py`: `AdaptDecomp` — main class running the per-batch whitening and source updates.
- `ops.py`: Pure tensor operations (KL divergence, NMS peak detection, centroid updates, orthonormalisation, etc.).
- `loaders.py`: Functions to load EMG and decomposition files into the format expected by `AdaptDecomp`.
- `preprocessing.py`: EMG preprocessing (bandpass filter, powerline removal, high-pass, low-pass).
- `optimize.py`: Optuna-based single- and multi-objective hyperparameter search.
- `io.py`: Functions to save and load adaptive decomposition outputs (HDF5).
- `plots.py`: Functions to visualise decomposition results.
- `utils.py`: Utility functions for motor unit properties and rate-of-agreement evaluation.

## Command line interface
The code is integrated with [Weights and Biases](https://wandb.ai) to track and visualise results. To run a single adaptive decomposition:

```sh
python scripts/run.py \
  --data_config configs/data_configs/data_example.yml \
  --model_config configs/model_configs/default_neuromotion.yml \
  --wandb_project_name adapt_decomp
```

To run a **wandb hyperparameter sweep** (random, grid, or Bayesian):
```sh
python scripts/run.py \
  --data_config configs/data_configs/data_example.yml \
  --model_config configs/model_configs/default_neuromotion.yml \
  --sweep_config configs/model_configs/sweep_loss.yml \
  --sweep_counts 30 \
  --wandb_project_name adapt_decomp
```

To run an **Optuna hyperparameter search** (self-contained, no wandb sweep agent required):
```sh
python scripts/run.py \
  --data_config configs/data_configs/data_example.yml \
  --model_config configs/model_configs/default_neuromotion.yml \
  --optim_config configs/model_configs/optim_params.yml \
  --n_trials 50 \
  --wandb_project_name adapt_decomp
```

The `data_config` is a YAML wrapper with paths to the calibrated decomposition model (`path_decomp`), the input EMG (`path_emg`), a preprocessing flag (`preprocess`), and the data loader name (`loader`). See `configs/data_configs/` for examples.

## Config reference

All active fields in `Config`. Legacy fields (`max_rel_delta_v`, `max_rel_delta_b`, `cov_alpha`, `contrast_fun`, etc.) are accepted by the YAML loader for backward compatibility but are not used by any logic.

| Field | Default | Description |
|-------|---------|-------------|
| `fs` | `2048` | Sampling frequency (Hz) |
| `device` | `null` | Compute device: `null` (auto-detect), `"cpu"`, `"cuda"`, `"mps"` |
| `lowcut` | `20.0` | Bandpass high-pass cutoff (Hz) |
| `highcut` | `500.0` | Bandpass low-pass cutoff (Hz) |
| `powerline` | `true` | Remove powerline artefact |
| `powerline_freq` | `50.0` | Powerline frequency (Hz) |
| `ext_fact` | `10` | Time-delay embedding factor; D = channels × ext_fact |
| `batch_ms` | `100` | Batch duration in ms; batch_size = batch_ms × fs / 1000 |
| `adapt_wh` | `true` | Enable whitening matrix wh adaptation |
| `adapt_sv` | `true` | Enable separation matrix sv adaptation |
| `adapt_sd` | `true` | Enable spike/base centroid adaptation |
| `compute_loss` | `true` | Log wh_loss, sv_loss, centroid_loss, wh_trace each batch |
| `wh_learning_rate` | `5e-3` | Whitening learning rate: applied step ≈ `wh_learning_rate · ‖wh‖ · e_wh` along the (unit-normalized) natural-gradient direction. **Reuses the v1 field name but NOT its formula** — v1 multiplied a raw gradient directly; here the direction is first unit-normalized. A value tuned against v1 does not carry over and must be re-tuned. |
| `sv_learning_rate` | `1e-3` | Separation-vector learning rate: applied step ≈ `sv_learning_rate · ‖sv_row‖ · e_sv` along the (unit-normalized) natural-gradient direction. Same v1-name-reuse caveat as `wh_learning_rate` above. |
| `safety_clip_multiplier_wh` | `20.0` | Rare safety-net ceiling for wh, expressed as a multiple of `wh_learning_rate` (not independently tunable — scales with `wh_learning_rate` so it can't collapse to always-on during search) |
| `safety_clip_multiplier_sv` | `20.0` | Rare safety-net ceiling for sv, expressed as a multiple of `sv_learning_rate` |
| `ema_alpha` | `0.95` | Smoothing rate for the EMA of ‖direction‖/‖grad_sv‖ used to normalize the natural-gradient direction to unit scale before scaling by `wh_learning_rate`/`sv_learning_rate` |
| `wh_mode` | `"kl_to_identity"` | KL divergence target: `"kl_to_identity"` or `"kl_to_cal"` |
| `wh_b_coupling` | `false` | Propagate wh-step frame correction to sv |
| `shrinkage` | `1e-3` | Ledoit-Wolf shrinkage applied to FIFO covariance |
| `fifo_length` | `0` | FIFO samples for covariance estimation (0 = auto: 2×D) |
| `max_sigma_batches` | `300` | Max calibration windows for sigma_K/kappa estimation (0 = all) |
| `contrast_scope` | `"batch_based"` | Contrast domain: `"batch_based"` or `"spike_based"` |
| `sv_epochs` | `1` | Fixed-point iterations per batch for sv |
| `sv_tol` | `1e-4` | Early-exit threshold for sv fixed-point loop |
| `spike_dist_ms` | `10` | Minimum inter-spike interval for NMS (ms) |
| `source_fifo_batches` | `2` | Past batches of sources prepended for left-edge NMS context |
| `centroid_momentum` | `0.95` | EMA momentum for centroid updates (0 = no memory) |
| `debug` | `false` | Store per-batch diagnostics in `outputs["diagnostics"]` |

## Data loaders
To use the command line interface with your own data, implement the corresponding data loader in [loaders.py](https://github.com/imendezguerra/adapt_decomp/blob/main/src/adapt_decomp/loaders.py) and add it to the `load_data` wrapper function. Also create the corresponding `.yml` file and store it under `configs/data_configs/`.

## Optimization
The default hyperparameters are optimised for the dataset in [Mendez Guerra et al, JNE, 2024](https://dx.doi.org/10.1088/1741-2552/ad5ebf) for simulations (neuromotion) and experimental wrist and forearm data. Two hyperparameter search methods are available:

**Wandb sweep** — random, grid, or Bayesian optimisation managed by the wandb agent. Requires a wandb account. See `configs/model_configs/sweep_loss.yml` for the parameter space format.

**Optuna** — self-contained single-objective search (no wandb sweep agent required). Results are logged to wandb as a run summary. See `configs/model_configs/optim_params.yml` for the parameter space format.

## Changes from v1

Key differences between `main` (v1) and `feature_sim` (v2):

| Aspect | v1 (`main`) | v2 (`feature_sim`) |
|--------|-------------|---------------------|
| Step-size control | `wh_learning_rate`, `sv_learning_rate` (scalar multipliers on a raw gradient) | Same field names, but reused for a different formula: applied to a unit-normalized natural-gradient direction instead (`step ≈ wh_learning_rate · ‖ref‖ · e`); `safety_clip_multiplier_{v,b}` sets a rare safety-net ceiling scaled to `wh_learning_rate`/`sv_learning_rate`, replacing the earlier `max_rel_delta_{v,b}` trust-region clip that was found to engage on ~100% of batches. **A value tuned against v1 does not carry over — re-tune from scratch, don't reuse the old number.** |
| Whitening update | Recursive EMA covariance + KL gradient | FIFO covariance + natural-gradient update; two KL modes (`kl_to_identity`, `kl_to_cal`) |
| Source update | Per-unit loop, scipy `find_peaks` | Vectorised NMS + adaptive centroids, GPU-compatible |
| Orthonormalisation | Gram-Schmidt deflation (per unit) | QR decomposition (all units at once, fixed); Gram-Schmidt still available as a standalone `ops.py` utility (`orthonormalize_rows_gram_schmidt`) for ablations, no longer Config-driven |
| Spike detection | scipy `find_peaks` with fixed height threshold | Vectorised max-pool NMS with adaptive thresholds and IQR outlier gate |
| Output keys | `wh_loss`, `sv_loss`, `total_loss` | + `centroid_loss`, `wh_trace`; all optional via `compute_loss` |
| Hyperparameter search | wandb sweep (wandb-managed) | wandb sweep + Optuna (self-contained); results optionally logged to wandb |

## Contributing
We welcome contributions! Here's how you can contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/newfeature`).
3. Commit your changes (`git commit -m 'Add some newfeature'`).
4. Push to the branch (`git push origin feature/newfeature`).
5. Open a pull request.

## License
This repository is licensed under the MIT License.

## Citation

If you use this code in your research, please cite this repository:

```
@article{Mendez Guerra_2024,
   author={Mendez Guerra, Irene and Barsakcioglu, Deren Y. and Farina, Dario},
   title={Adaptive EMG decomposition in dynamic conditions based on online learning metrics with tunable hyperparameters},
   journal={Journal of Neural Engineering},
   publisher={IOP Publishing},
   volume={21},
   number={4},
   ISSN={1741-2552},
   DOI={10.1088/1741-2552/ad5ebf},
   url={https://dx.doi.org/10.1088/1741-2552/ad5ebf}
   }
```

## Contact

For any questions or inquiries, please contact us at:
```
Irene Mendez Guerra
irene.mendez17@imperial.ac.uk
```
